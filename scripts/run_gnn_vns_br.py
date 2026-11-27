"""GNN-VNS-BR training, three ranking controls and fixed-budget experiments."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import io
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np
import torch
from numba import set_num_threads

from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.fixed_taskset import generate_taskset
from mrta_reference.model import ParentWeld, ScheduleStatus
from mrta_search.gnn_vns_br import (
    ACTION_FEATURES, EDGE_FEATURES, EDGE_RELATIONS, MODEL_VARIANTS, NODE_FEATURES, ROBOT_FEATURES,
    BRConfig, FeatureConfig, FixedWorkspace,
    candidate_actions, local_graph, make_ranker, ranking_pairs, run_gnn_vns_br, train_ranker,
)
from run_fixed_taskset_experiment import (
    CONFIG, FIELDS, LIMIT, REPORT, RESULTS, SEEDS, json_text, load_tasksets,
    read_rows as read_baselines, solution_record,
)

WEIGHTS = ROOT / 'models/gnn_vns_br_atomic.pt'
CSV = ROOT / 'results/native_priority/new_dataset_final/gnn_vns_br_atomic.csv'
SCALE_WEIGHTS = ROOT / 'models/gnn_vns_br_atomic_scale3000.pt'
SCALE_CSV = ROOT / 'results/native_priority/new_dataset_final/gnn_vns_br_atomic_scale3000.csv'
SCALE_DATA = ROOT / 'data/weld_layout/scale1000'
OUTPUT_FIELDS = ('trial', *FIELDS)
SCALE_FIELDS = ('dataset', 'variant', *OUTPUT_FIELDS)
SECTION = '\n## GNN-VNS-BR 边特征与搜索改进（TRAIN Smoke）\n'
SCALE_SECTION = '\n## GNN-VNS-BR scale1000训练、消融与正式实验\n'


def iter_tasksets(split, dataset='original'):
    """Stream new layouts through the unchanged fixed splitter; no data copy."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError(f'unknown split: {split}')
    if dataset not in ('original', 'scale1000'):
        raise ValueError(f'unknown dataset: {dataset}')
    base = SCALE_DATA if dataset == 'scale1000' else ROOT / 'data/weld_layout'
    with (base / f'{split}.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            parents = tuple(ParentWeld(p['parent_id'], tuple(p['start']), tuple(p['end']))
                            for p in record['parents'])
            yield record['layout_id'], generate_taskset(parents, CONFIG)


def training_tasksets(dataset='original'):
    original = {name: ts for name, ts in iter_tasksets('train')}
    yield from original.items()
    if dataset == 'original':
        return
    for name, taskset in iter_tasksets('train', dataset):
        if name in original:
            if taskset != original[name]:
                raise ValueError(f'scale TRAIN differs from the original layout with the same key: {name}')
            continue
        yield name, taskset


def fit_training_normalization(dataset='scale1000'):
    """One global reference fitted from TRAIN geometry, without scheduling."""
    loads = [sum(CONFIG.process_time(t.length) for t in ts.tasks) / 4.
             for _, ts in training_tasksets(dataset)]
    if not loads:
        raise ValueError('TRAIN is empty')
    return FeatureConfig(time_ref=max(1., median(loads)))


def read_results(dataset='original'):
    path = SCALE_CSV if dataset == 'scale1000' else CSV
    if not path.exists():
        return []
    csv.field_size_limit(LIMIT)
    text = path.read_text(encoding='utf-8')
    if text and not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n'
    return list(csv.DictReader(io.StringIO(text)))


def append_result(row, dataset='original'):
    path = SCALE_CSV if dataset == 'scale1000' else CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = path.read_bytes() if path.exists() else b''
    if saved and not saved.endswith(b'\n'):
        # A stopped append may leave only its final physical line incomplete.
        saved = saved[:saved.rfind(b'\n') + 1]
        with path.open('r+b') as stream:
            stream.truncate(len(saved))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=SCALE_FIELDS if dataset == 'scale1000' else OUTPUT_FIELDS)
    if not saved:
        writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if len(saved) + len(text.encode('utf-8')) >= LIMIT:
        raise ValueError('GNN-VNS-BR CSV would exceed 100MB')
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()


def build_supervision(*, seed=20261126, states_per_source=2, settings=BRConfig(),
                      dataset='original', normalization=None, progress=print):
    """Use TRAIN teachers, independent greedy starts, and one legal perturbation.

    Candidate budgets and graph construction are identical to online search.
    Labels stay in memory; no validation/test state is turned into a label.
    """
    rng = random.Random(seed)
    normalization = normalization or fit_training_normalization(dataset)
    tasksets = training_tasksets(dataset)
    teachers = {}
    for method in ('wag_vns', 'hga'):
        for row in read_baselines(method, historical=True):
            if (row['split'] != 'train' or row['termination_reason'] != 'ITERATION_LIMIT'
                    or not row['solution'] or not row.get('native_cmax_s')
                    or not math.isfinite(float(row['native_cmax_s']))):
                continue
            key = row['layout_id'], method
            old = teachers.get(key)
            if old is None or float(row['native_cmax_s']) < float(old['native_cmax_s']):
                teachers[key] = row
    examples = []
    stats = dict(source_split='train', dataset=dataset, teacher_solutions=0, greedy_initializations=0,
                 generated_states=0, states=0, candidate_labels=0, feasible_labels=0,
                 deadlock_labels=0, structural_infeasible_labels=0, numeric_failure_labels=0,
                 pairs=0, duplicate_states_skipped=0, uninformative_states=0,
                 layouts=[], state_sources={'wag_vns':0, 'hga':0, 'greedy':0},
                 skipped_state_statuses={'INFEASIBLE':0, 'NUMERIC_FAILURE':0},
                 perturbed_states=0, candidate_parameters=asdict(settings))
    started = time.perf_counter()
    for name, taskset in tasksets:
        workspace = FixedWorkspace(taskset, CONFIG, normalization=normalization)
        identities = set()
        sources = []
        for method in ('wag_vns', 'hga'):
            row = teachers.get((name, method))
            if row is not None:
                sources.append((method, workspace.restore(json.loads(row['solution']))))
                stats['teacher_solutions'] += 1
        greedy = workspace.initialize(rng, settings, time.perf_counter() + settings.time_limit)
        if greedy is not None:
            sources.append(('greedy', greedy))
            stats['greedy_initializations'] += 1
        for source, initial in sources:
            current = initial
            for perturbation in range(states_per_source):
                if current.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                    stats['skipped_state_statuses'][current.status.value] += 1
                    break
                actions = candidate_actions(workspace, current, settings, rng)
                if perturbation and actions:
                    action = rng.choice(actions)
                    current = workspace.evaluate(action.routes, action.bits)
                    if current.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                        stats['skipped_state_statuses'][current.status.value] += 1
                        break
                    actions = candidate_actions(workspace, current, settings, rng)
                identity = tuple((tuple(r), tuple(d)) for r, d in zip(current.routes, current.bits))
                if identity in identities:
                    stats['duplicate_states_skipped'] += 1
                    continue
                identities.add(identity)
                stats['generated_states'] += 1
                if not actions:
                    continue
                evaluated = [workspace.evaluate(a.routes, a.bits) for a in actions]
                pairs = ranking_pairs(evaluated, rng)
                stats['candidate_labels'] += len(evaluated)
                stats['feasible_labels'] += sum(s.feasible for s in evaluated)
                stats['deadlock_labels'] += sum(s.status is ScheduleStatus.DEADLOCK for s in evaluated)
                stats['structural_infeasible_labels'] += sum(s.status is ScheduleStatus.INFEASIBLE for s in evaluated)
                stats['numeric_failure_labels'] += sum(s.status is ScheduleStatus.NUMERIC_FAILURE for s in evaluated)
                if not len(pairs):
                    stats['uninformative_states'] += 1
                    continue
                examples.append((name, local_graph(workspace, current, actions), pairs))
                stats['states'] += 1
                stats['state_sources'][source] += 1
                stats['perturbed_states'] += int(perturbation > 0)
                stats['pairs'] += len(pairs)
        stats['layouts'].append(name)
        progress('BR_LABELS', name, {k:stats[k] for k in ('states','candidate_labels','pairs')},
                 'elapsed_s', round(time.perf_counter() - started, 3))
    stats['label_generation_s'] = time.perf_counter() - started
    stats['layout_records'] = len(stats['layouts'])
    stats['unique_layouts'] = len(set(stats['layouts']))
    if not examples:
        raise RuntimeError('no informative TRAIN labels were generated')
    return examples, stats


def train(args):
    path = args.weights or (SCALE_WEIGHTS if args.dataset == 'scale1000' else WEIGHTS)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('model outputs must stay inside DRL')
    if path.exists():
        check_model_features(torch.load(path, map_location='cpu', weights_only=False))
        print('BR_TRAIN_ALREADY_SAVED', str(path), flush=True)
        return
    started = time.perf_counter()
    settings = BRConfig()
    normalization = fit_training_normalization(args.dataset)
    examples, data = build_supervision(seed=args.training_seed, settings=settings,
                                      dataset=args.dataset, normalization=normalization)
    variants = (args.variant,)
    models, learning = {}, {}
    for variant in variants:
        tick = time.perf_counter()
        model, record = train_ranker(examples, seed=args.training_seed, epochs=args.epochs, variant=variant)
        record['training_s'] = time.perf_counter() - tick
        models[variant], learning[variant] = model.state_dict(), record
        print('BR_MODEL_TRAINED', variant, json_text(record), flush=True)
    training = dict(data=data, seed=args.training_seed, epochs=args.epochs,
                    total_offline_s=time.perf_counter() - started,
                    training_s=sum(record['training_s'] for record in learning.values()),
                    validation_used_for_labels=False, test_used_for_labels=False,
                    shared_supervision=True)
    checkpoint = dict(training=training, settings=asdict(settings), node_features=NODE_FEATURES,
                      robot_features=ROBOT_FEATURES, edge_relations=EDGE_RELATIONS,
                      normalization=asdict(normalization),
                      edge_features=EDGE_FEATURES, action_features=ACTION_FEATURES)
    if args.dataset == 'scale1000':
        checkpoint.update(models=models, learning=learning)
    else:
        checkpoint['model'] = models[args.variant]
        checkpoint['variant'] = args.variant
        training['learning'] = learning[args.variant]
    # The one approved weight file contains the model and its actual training
    # record, with no optimizer, duplicate snapshots or sample sidecars.
    payload = io.BytesIO()
    torch.save(checkpoint, payload)
    if payload.tell() >= LIMIT:
        raise ValueError('model would exceed 100MB')
    with path.open('xb') as stream:
        stream.write(payload.getvalue())
    print('BR_TRAINED', json_text({k: v for k, v in training.items() if k != 'learning'}), flush=True)
    (update_scale_report if args.dataset == 'scale1000' else update_report)()


def check_model_features(checkpoint):
    if (tuple(checkpoint.get('node_features', ())) != NODE_FEATURES
            or tuple(checkpoint.get('robot_features', ())) != ROBOT_FEATURES
            or tuple(checkpoint.get('edge_relations', ())) != EDGE_RELATIONS
            or tuple(checkpoint.get('edge_features', ())) != EDGE_FEATURES
            or tuple(checkpoint.get('action_features', ())) != ACTION_FEATURES
            or 'normalization' not in checkpoint):
        raise ValueError('incompatible checkpoint: atomic heterograph requires '
                         f'{len(NODE_FEATURES)} node/{len(EDGE_FEATURES)} edge/{len(ACTION_FEATURES)} action features; '
                         'historical weights are preserved, not adapted or overwritten')
    FeatureConfig(**checkpoint['normalization'])


def load_model(dataset='original', variant='full_gnn', *, checkpoint=None, path=None):
    if checkpoint is None:
        checkpoint = torch.load(path or (SCALE_WEIGHTS if dataset == 'scale1000' else WEIGHTS),
                                 map_location='cpu', weights_only=False)
    check_model_features(checkpoint)
    if dataset == 'original' and checkpoint.get('variant', 'full_gnn') != variant:
        raise ValueError('requested ranking variant differs from the saved model')
    training = checkpoint.get('training', {})
    learning = (checkpoint.get('learning', {}).get(variant, {}) if dataset == 'scale1000'
                else training.get('learning', {}))
    if training.get('epochs', 0) < 1 or learning.get('selected_epoch', 0) < 1:
        raise ValueError('checkpoint has no completed training record')
    model = make_ranker(variant, normalization=FeatureConfig(**checkpoint['normalization']))
    model.load_state_dict(checkpoint['models'][variant] if dataset == 'scale1000' else checkpoint['model'])
    model.eval(); model.trained = True
    return model, checkpoint


def run_one(split, name, taskset, seed, *, trial='main', time_limit=None, model=None, checkpoint=None,
            dataset='original', source='original', variant='full_gnn'):
    if dataset == 'original' and split != 'train':
        raise ValueError('this round runs TRAIN smoke only; VALIDATION and TEST are untouched')
    started = time.perf_counter()
    if model is None:
        model, checkpoint = load_model(dataset, variant)
    else:
        check_model_features(checkpoint)
    settings = BRConfig(**checkpoint['settings'])
    if time_limit is not None:
        settings = replace(settings, time_limit=time_limit)
    workspace, state, diagnostic = run_gnn_vns_br(taskset, model, seed=seed, settings=settings, config=CONFIG)
    solution = workspace.solution(state)
    schedule = workspace.schedule_result(state)  # Reuse native arrays; no final scheduling call.
    learning = checkpoint['learning'][variant] if dataset == 'scale1000' else checkpoint['training']['learning']
    diagnostic.update(final_schedule_reused=True, final_scheduler_extra_calls=0,
                      formal_diagnostics=schedule.diagnostics,
                      robot_completion=schedule.robot_completion, waiting_s=schedule.waiting_time,
                      robot_waiting_s=schedule.robot_waiting_time,
                      native_feasible=state.feasible, selected_epoch=learning['selected_epoch'],
                      offline_training_s=checkpoint['training']['total_offline_s'],
                      input_adapter='FIXED_ATOMIC_HETEROGAT_NATIVE_PRIORITY', evaluation_policy='NATIVE_PRIORITY_FOUR_ROBOTS',
                      single_thread=True, cold_jit_included=True)
    method = f'gnn_vns_br_{variant}_atomic_scale3000' if dataset == 'scale1000' else 'gnn_vns_br_atomic'
    row = dict(trial=trial, split=split, layout_id=name, seed=seed, method=method,
               status=schedule.status.value, cmax_s=schedule.cmax if schedule.feasible else None,
               native_cmax_s=state.value if state.feasible else None,
               runtime_s=time.perf_counter() - started, iterations=diagnostic['rounds'],
               termination_reason=diagnostic['termination_reason'], coordination_calls=workspace.evaluations,
               parameters=json_text(asdict(settings)), diagnostic=json_text(diagnostic),
               solution=json_text(solution_record(solution, state.bits)))
    if dataset == 'scale1000':
        row.update(dataset=source, variant=variant)
    print('BR_RESULT', json_text({k: v for k, v in row.items() if k not in ('solution', 'diagnostic')}), flush=True)
    return row








def summarize_group(rows, *, source, split, variant):
    group = [r for r in rows if r['dataset'] == source and r['split'] == split
             and r['variant'] == variant and r['trial'] == 'main']
    costs = [float(r['cmax_s']) for r in group if r['status'] == 'FEASIBLE' and r['cmax_s']]
    times = [float(r['runtime_s']) for r in group]
    details = [json.loads(r['diagnostic']) for r in group]
    improvements = []
    for row, detail in zip(group, details):
        initial = detail.get('initial_native_cmax_s')
        if initial and row['native_cmax_s']:
            improvements.append(1. - float(row['native_cmax_s']) / initial)
    return dict(dataset=source, split=split, variant=variant, runs=len(group), feasible=len(costs),
                failures={status:sum(r['status'] == status for r in group)
                          for status in sorted({r['status'] for r in group}) if status != 'FEASIBLE'},
                cmax_mean_s=mean(costs) if costs else None,
                cmax_median_s=median(costs) if costs else None,
                runtime_mean_s=mean(times) if times else None,
                runtime_median_s=median(times) if times else None,
                initial_improvement_pairs=len(improvements),
                initial_native_improvement_mean=mean(improvements) if improvements else None,
                initial_native_improvement_median=median(improvements) if improvements else None,
                native_feasible=sum(d['native_feasible'] for d in details),
                initial_native_feasible=sum(d.get('initial_native_cmax_s') is not None for d in details),
                rounds_median=median([int(r['iterations']) for r in group]) if group else None,
                accepted_median=median([d['accepted'] for d in details]) if group else None,
                generated_by_move=[sum(d.get('generated_by_move', [0]*6)[m] for d in details) for m in range(6)],
                evaluated_by_move=[sum(d.get('evaluated_by_move', [0]*6)[m] for d in details) for m in range(6)],
                feasible_by_move=[sum(d.get('feasible_by_move', [0]*6)[m] for d in details) for m in range(6)])


def paired_comparison(rows, split='validation'):
    group = [r for r in rows if r['dataset'] == 'original' and r['split'] == split
             and r['variant'] == 'full_gnn' and r['trial'] == 'main']
    result = {}
    for method in ('wag_vns', 'hga'):
        baseline = {(r['layout_id'], int(r['seed'])):r for r in read_baselines(method)
                    if r['split'] == split and r['termination_reason'] == 'ITERATION_LIMIT'}
        differences = []
        for row in group:
            other = baseline.get((row['layout_id'], int(row['seed'])))
            if (other is not None and row['status'] == other['status'] == 'FEASIBLE'
                    and row['cmax_s'] and other['cmax_s']):
                differences.append(float(row['cmax_s']) / float(other['cmax_s']) - 1.)
        durations = [float(r['runtime_s']) for r in baseline.values()]
        result[method] = dict(pairs=len(differences),
                              median_relative_cmax=median(differences) if differences else None,
                              mean_relative_cmax=mean(differences) if differences else None,
                              baseline_runs=len(baseline),
                              baseline_feasible=sum(r['status'] == 'FEASIBLE' for r in baseline.values()),
                              baseline_runtime_median_s=median(durations) if durations else None)
    return result


def validation_admission(rows):
    summary = summarize_group(rows, source='original', split='validation', variant='full_gnn')
    paired = paired_comparison(rows)
    expected = {(name, seed) for name, _ in iter_tasksets('validation') for seed in SEEDS}
    actual = {(r['layout_id'], int(r['seed'])) for r in rows if r['dataset'] == 'original'
              and r['split'] == 'validation' and r['variant'] == 'full_gnn' and r['trial'] == 'main'}
    wag, hga = paired['wag_vns'], paired['hga']
    runtime = summary['runtime_median_s']
    checks = dict(complete_30=len(expected) == 30 and actual == expected and summary['runs'] == 30,
                  coordination_feasible_28=summary['feasible'] >= 28,
                  paired_wag_improves=wag['median_relative_cmax'] is not None and wag['median_relative_cmax'] < 0.,
                  paired_hga_at_most_2pct=hga['median_relative_cmax'] is not None and hga['median_relative_cmax'] <= .02,
                  runtime_vs_wag=runtime is not None and wag['baseline_runtime_median_s'] is not None
                                  and runtime <= 1.5 * wag['baseline_runtime_median_s'],
                  runtime_vs_hga=runtime is not None and hga['baseline_runtime_median_s'] is not None
                                  and runtime < hga['baseline_runtime_median_s'])
    return dict(ready_for_test=all(checks.values()), checks=checks, summary=summary, paired=paired)


def run_experiment(split, *, variant='full_gnn', weights=None):
    rows = read_results('scale1000')
    if split == 'test':
        admission = validation_admission(rows)
        if not admission['ready_for_test']:
            print('BR_TEST_NOT_RUN', json_text(admission), flush=True)
            update_scale_report(result_rows=rows)
            return
    elif split != 'validation':
        raise ValueError('formal experiments use validation or test')
    models = {}
    checkpoint = None
    variants = (variant,) if split == 'validation' else ('full_gnn',)
    for variant in variants:
        models[variant], checkpoint = load_model('scale1000', variant, checkpoint=checkpoint, path=weights)
    keys = [(r['dataset'], r['split'], r['layout_id'], int(r['seed']), r['variant'], r['trial']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate saved model/layout/seed result keys')
    completed = {key for key, row in zip(keys, rows)
                 if int(row['coordination_calls']) > 0 and row['solution']
                 and row['status'] in {status.value for status in ScheduleStatus}}
    saved = 0
    for source in ('original', 'scale1000'):
        source_variants = variants if source == 'original' else ('full_gnn',)
        seeds = SEEDS if source == 'original' else SEEDS[:1]
        for name, taskset in iter_tasksets(split, source):
            for seed in seeds:
                for variant in source_variants:
                    key = source, split, name, seed, variant, 'main'
                    if key in completed:
                        print('BR_REUSED', *key, flush=True)
                        continue
                    print('BR_START', *key, flush=True)
                    row = run_one(split, name, taskset, seed, dataset='scale1000', source=source,
                                   variant=variant, model=models[variant], checkpoint=checkpoint)
                    append_result(row, 'scale1000')
                    rows.append(row); completed.add(key); saved += 1
                    if saved % 10 == 0:
                        update_scale_report(checkpoint=checkpoint, result_rows=rows)
    update_scale_report(checkpoint=checkpoint, result_rows=rows)
    print('BR_EXPERIMENT_DONE', split, 'new_rows', saved, flush=True)
    if split == 'validation':
        print('BR_VALIDATION_ADMISSION', json_text(validation_admission(rows)), flush=True)


def update_scale_report(*, checkpoint=None, result_rows=None):
    """Print native-policy summaries; historical and migration reports are read-only."""
    rows = read_results('scale1000') if result_rows is None else result_rows
    summaries = [summarize_group(rows, source=source, split=split, variant=variant)
                 for source in ('original', 'scale1000') for split in ('validation', 'test')
                 for variant in MODEL_VARIANTS]
    summary=dict(evaluation_policy='NATIVE_PRIORITY_FOUR_ROBOTS',
                 summaries=[item for item in summaries if item['runs']],
                 admission=validation_admission(rows))
    print('BR_NATIVE_SUMMARY',json_text(summary),flush=True)
    return summary


def update_readme():
    """Current-policy summaries stay in the native report; history is unchanged."""
    return update_scale_report()


def update_report(*, checkpoint=None, result_rows=None):
    """Print native smoke summaries without writing historical files."""
    rows = read_results() if result_rows is None else result_rows
    summary=[dict(layout_id=r['layout_id'],seed=r['seed'],status=r['status'],
                  cmax_s=r.get('cmax_s'),runtime_s=r['runtime_s']) for r in rows]
    print('BR_NATIVE_SMOKE_SUMMARY',json_text(summary),flush=True)
    return summary


def prepare(args):
    """One TRAIN layout: interface checks only, with no labels or weight writes."""
    normalization = fit_training_normalization(args.dataset)
    name, taskset = next(iter(training_tasksets(args.dataset)))
    workspace = FixedWorkspace(taskset, CONFIG, normalization=normalization)
    settings = BRConfig(candidates=4)
    rng = random.Random(args.training_seed)
    state = workspace.initialize(rng, settings, time.perf_counter() + settings.time_limit)
    if state is None:
        raise RuntimeError('no initial TRAIN state')
    actions = candidate_actions(workspace, state, settings, rng)
    graph = local_graph(workspace, state, actions)
    if not actions:
        raise RuntimeError('the selected TRAIN state has no legal actions')
    evaluations = workspace.evaluations
    model = make_ranker(normalization=normalization)
    scores = model(graph)
    scores.square().mean().backward()
    finite = bool(torch.isfinite(scores).all() and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    if not finite or len(graph.nodes) != len(taskset.tasks) + 4:
        raise RuntimeError('invalid graph, outputs or gradients')
    if workspace.evaluations != evaluations:
        raise RuntimeError('graph construction/forward unexpectedly scheduled a candidate')
    print('BR_PREPARED', json_text(dict(layout_id=name, split='train',
        atomic_tasks=len(taskset.tasks), nodes=list(graph.nodes.shape),
        edge_relations=[EDGE_RELATIONS[i] for i in sorted(set(graph.relations.tolist()))],
        edges=list(graph.edge_attr.shape), actions=list(graph.action_features.shape),
        finite_outputs_and_gradients=finite, normalization=asdict(normalization),
        initialization=workspace.initialization, native_evaluations=evaluations,
        graph_scheduler_calls=0, labels_generated=0, optimizer_steps=0, files_written=0)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'validation', 'test', 'summary', 'smoke'))
    parser.add_argument('--dataset', choices=('original', 'scale1000'), default='scale1000')
    parser.add_argument('--training-seed', type=int, default=20261126)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--variant', choices=MODEL_VARIANTS, default='full_gnn')
    parser.add_argument('--weights', type=Path, help='new atomic-model checkpoint; historical weights are incompatible')
    parser.add_argument('--layouts', nargs='+', default=('train_01','train_12','train_30'))
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--trial', default='main')
    args = parser.parse_args()
    set_num_threads(1); torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'train':
        train(args)
    elif args.stage in ('validation', 'test'):
        if args.dataset != 'scale1000':
            raise ValueError('formal stages use the new scale1000 checkpoint; old experiments are preserved')
        run_experiment(args.stage, variant=args.variant, weights=args.weights)
    elif args.stage == 'smoke':
        if args.dataset != 'original':
            raise ValueError('the previous TRAIN smoke uses --dataset original; this round uses validation')
        model, checkpoint = load_model(variant=args.variant, path=args.weights)
        tasksets = {name:ts for split,name,ts in load_tasksets() if split=='train'}
        if len(args.layouts) != len(set(args.layouts)) or any(name not in tasksets for name in args.layouts):
            raise ValueError('smoke layouts must be distinct existing TRAIN layouts')
        existing = read_results()
        for name in args.layouts:
            previous = next((r for r in existing if r['trial']==args.trial and r['split']=='train'
                             and r['layout_id']==name and int(r['seed'])==args.seed), None)
            if previous is not None:
                print('BR_SMOKE_REUSED',name,args.seed,previous['status'],previous['cmax_s'],flush=True)
                continue
            row = run_one('train',name,tasksets[name],args.seed,trial=args.trial,model=model,checkpoint=checkpoint)
            append_result(row)
            existing.append(row)
            update_report(checkpoint=checkpoint,result_rows=existing)
    else:
        if args.dataset == 'scale1000':
            update_scale_report()
            update_readme()
            print('BR_VALIDATION_ADMISSION', json_text(validation_admission(read_results('scale1000'))), flush=True)
        else:
            update_report()
            for row in read_results():
                print('BR_SMOKE_SUMMARY',json_text({k:v for k,v in row.items() if k not in ('solution','diagnostic')}),flush=True)


if __name__ == '__main__':
    main()
