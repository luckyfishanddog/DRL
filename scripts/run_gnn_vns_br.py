"""GNN-VNS-BR training, three ranking controls and fixed-budget experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import csv
from dataclasses import asdict, fields, replace
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
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
import torch
from numba import get_num_threads, set_num_threads

from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.fixed_taskset import generate_taskset
from mrta_reference.model import ParentWeld, ScheduleStatus
from mrta_search.gnn_vns_br import (
    ACTION_FEATURES, EDGE_FEATURES, EDGE_RELATIONS, MODEL_VARIANTS, NODE_FEATURES, ROBOT_FEATURES,
    BRConfig, FeatureConfig, FixedWorkspace, LocalGraph,
    candidate_actions, local_graph, make_ranker, ranking_pairs, run_gnn_vns_br, train_ranker,
)
from run_fixed_taskset_experiment import (
    CONFIG, FIELDS, LIMIT, REPORT, RESULTS, SEEDS, json_text, load_tasksets,
    read_rows as read_baselines, solution_record,
    TEACHER_RESULTS, feasible_teacher, teacher_coverage, scale3000_tasksets, hga_training_selection,
    restore_solution,
)

WEIGHTS = ROOT / 'models/gnn_vns_br_atomic_16d.pt'
CSV = ROOT / 'results/native_priority/new_dataset_final/gnn_vns_br_atomic_16d.csv'
SCALE_WEIGHTS = ROOT / 'models/gnn_vns_br_atomic_16d_scale3000.pt'
SCALE_CSV = ROOT / 'results/native_priority/new_dataset_final/gnn_vns_br_atomic_16d_scale3000.csv'
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
    original = {}
    for name, taskset in iter_tasksets('train'):
        if name in original:
            raise ValueError(f'duplicate original TRAIN layout: {name}')
        original[name] = taskset
    yield from original.items()
    if dataset == 'original':
        return
    seen = set()
    for name, taskset in iter_tasksets('train', dataset):
        if name in seen:
            raise ValueError(f'duplicate scale TRAIN layout: {name}')
        seen.add(name)
        if name in original:
            if taskset != original[name]:
                raise ValueError(f'scale TRAIN differs from the original layout with the same key: {name}')
            continue
        yield name, taskset


def fit_training_normalization(dataset='scale1000', *, tasksets=None):
    """One global reference fitted from TRAIN geometry, without scheduling."""
    loads = [sum(CONFIG.process_time(t.length) for t in ts.tasks) / 4.
             for _, ts in (training_tasksets(dataset) if tasksets is None else tasksets)]
    if not loads:
        raise ValueError('TRAIN is empty')
    return FeatureConfig(time_ref=max(1., median(loads)))


def selected_training_tasksets(dataset='original', layouts=None):
    """Validate an optional TRAIN-only selection before generating any labels."""
    if layouts is None:
        yield from training_tasksets(dataset)
        return
    requested = tuple(layouts)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('TRAIN layout selection must be nonempty and contain no duplicates')
    selected = {}
    for name, taskset in training_tasksets(dataset):
        if name in requested:
            if name in selected:
                raise ValueError(f'duplicate TRAIN layout: {name}')
            selected[name] = taskset
    missing = sorted(set(requested) - selected.keys())
    if missing:
        raise ValueError(f'unknown TRAIN layouts: {missing}')
    yield from ((name, selected[name]) for name in requested)


def read_training_teachers(layouts=None):
    """Read completed saved teachers; never substitute another Cmax field."""
    teachers = {}
    for method in ('wag_vns', 'hga'):
        rows = [*read_baselines(method, historical=True), *read_baselines(method),
                *read_baselines(method, directory=TEACHER_RESULTS)]
        for row in rows:
            if (not feasible_teacher(row,method)
                    or (layouts is not None and row.get('layout_id') not in layouts)):
                continue
            value = float(row['native_cmax_s'])
            key = row['layout_id'], method
            old = teachers.get(key)
            if old is None or value < float(old['native_cmax_s']):
                teachers[key] = row
    return teachers


def check_training_teachers(tasksets=None, original_ids=None):
    """Validate full TRAIN coverage before creating labels or training a model."""
    if tasksets is None:
        tasksets, original_ids = scale3000_tasksets()
    names = {name for name,_ in tasksets}
    if len(tasksets) != 3000 or len(names) != 3000 or original_ids is None or len(original_ids) != 40:
        raise ValueError('full training requires 3000 unique TRAIN layouts including the original 40')
    targets = dict(wag_vns=names,hga=hga_training_selection(tasksets,original_ids))
    by_name = dict(tasksets)
    summaries, invalid = {}, {}
    for method,selected in targets.items():
        completed, summary = teacher_coverage(method,selected)
        summary['original_completed'] = len(original_ids & completed.keys())
        summary['invalid_feasible_teachers'] = sorted(name for name,row in completed.items()
                if row['status']=='FEASIBLE' and not feasible_teacher(row,method))
        for name,row in completed.items():
            if row['status'] != 'FEASIBLE' or name in summary['invalid_feasible_teachers']:
                continue
            try:
                # Existing canonicalization checks atom coverage/assignment;
                # this is structural validation, with no scheduling call.
                restore_solution(by_name[name],row)
            except (ValueError,TypeError,KeyError,IndexError):
                summary['invalid_feasible_teachers'].append(name)
        summary['invalid_feasible_teachers'].sort()
        summaries[method] = summary
        print('BR_TEACHER_COVERAGE',json_text(summary),flush=True)
        if summary['missing'] or summary['duplicate_complete_keys'] or summary['invalid_feasible_teachers']:
            invalid[method] = {k:summary[k] for k in ('missing','duplicate_complete_keys','invalid_feasible_teachers')}
    if invalid:
        raise ValueError(f'incomplete or invalid TRAIN teacher coverage; no labels/training started: {json_text(invalid)}')
    return summaries


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


def _supervision_stats(dataset, layouts, settings):
    stats = dict(source_split='train', dataset=dataset, teacher_solutions=0, greedy_initializations=0,
                 generated_states=0, states=0, candidate_labels=0, feasible_labels=0,
                 deadlock_labels=0, structural_infeasible_labels=0, numeric_failure_labels=0,
                 pairs=0, duplicate_states_skipped=0, uninformative_states=0,
                 layouts=[], state_sources={'wag_vns':0, 'hga':0, 'greedy':0},
                 skipped_state_statuses={'INFEASIBLE':0, 'NUMERIC_FAILURE':0},
                 perturbed_states=0, candidate_parameters=asdict(settings))
    stats.update(requested_layouts=list(layouts) if layouts is not None else None,
                 teacher_solutions_by_method={'wag_vns': 0, 'hga': 0},
                 skipped_teachers={'wag_vns': 0, 'hga': 0},
                 teacher_state_statuses={status.value: 0 for status in ScheduleStatus},
                 native_evaluations=0, candidate_native_evaluations=0)
    return stats


def _build_supervision_serial(*, seed=20261126, states_per_source=2, settings=BRConfig(),
                              dataset='original', normalization=None, layouts=None, progress=print,
                              tasksets=None, teachers=None, allow_empty=False):
    """The existing label algorithm, run locally on supplied TRAIN layouts."""
    rng = random.Random(seed)
    normalization = normalization or fit_training_normalization(dataset)
    tasksets = selected_training_tasksets(dataset, layouts) if tasksets is None else tasksets
    if layouts is not None:
        tasksets = list(tasksets)
    teachers = read_training_teachers(layouts) if teachers is None else teachers
    examples = []
    stats = _supervision_stats(dataset, layouts, settings)
    started = time.perf_counter()
    for name, taskset in tasksets:
        workspace = FixedWorkspace(taskset, CONFIG, normalization=normalization)
        identities = set()
        sources = []
        for method in ('wag_vns', 'hga'):
            row = teachers.get((name, method))
            if row is not None:
                try:
                    restored = workspace.restore(json.loads(row['solution']))
                except (ValueError, TypeError, KeyError, IndexError):
                    stats['skipped_teachers'][method] += 1
                    continue
                stats['teacher_state_statuses'][restored.status.value] += 1
                if restored.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                    stats['skipped_teachers'][method] += 1
                    continue
                sources.append((method, restored))
                stats['teacher_solutions'] += 1
                stats['teacher_solutions_by_method'][method] += 1
        # Offline labels use the same bounded construction attempts. Do not
        # let worker startup/JIT contention change which attempts are made.
        greedy = workspace.initialize(rng, settings, math.inf)
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
                before = workspace.evaluations
                evaluated = [workspace.evaluate(a.routes, a.bits) for a in actions]
                stats['candidate_native_evaluations'] += workspace.evaluations - before
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
        stats['native_evaluations'] += workspace.evaluations
        if progress is not None:
            progress('BR_LABELS', name, {k:stats[k] for k in ('states','candidate_labels','pairs')},
                     'elapsed_s', round(time.perf_counter() - started, 3))
    stats['label_generation_s'] = time.perf_counter() - started
    stats['layout_records'] = len(stats['layouts'])
    stats['unique_layouts'] = len(set(stats['layouts']))
    if not examples and not allow_empty:
        raise RuntimeError('no informative TRAIN labels were generated')
    return examples, stats


@contextmanager
def _torch_threads(count):
    if count < 1:
        raise ValueError('training threads must be positive')
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _initialize_label_worker():
    set_num_threads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _pack_label_examples(examples):
    """Transfer ordinary NumPy arrays, avoiding many shared Tensor handles."""
    packed = []
    for name,graph,pairs in examples:
        values = {}
        for field in fields(graph):
            value = getattr(graph,field.name)
            values[field.name] = value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else value
        packed.append((name,values,pairs.detach().cpu().numpy().copy()))
    return packed


def _unpack_label_examples(packed):
    return [(name,LocalGraph(**{key:torch.from_numpy(value) if isinstance(value,np.ndarray) else value
                              for key,value in values.items()}),torch.from_numpy(pairs))
            for name,values,pairs in packed]


def _label_layout_job(item):
    name,taskset,seed,states_per_source,settings,dataset,normalization,teachers = item
    examples,stats = _build_supervision_serial(seed=seed,states_per_source=states_per_source,
        settings=settings,dataset=dataset,normalization=normalization,tasksets=[(name,taskset)],
        teachers=teachers,progress=None,allow_empty=True)
    stats.update(worker_pid=os.getpid(),worker_numba_threads=get_num_threads(),
                 worker_torch_threads=torch.get_num_threads())
    return name,_pack_label_examples(examples),stats


def _merge_label_stats(total, detail):
    for key in ('teacher_solutions','greedy_initializations','generated_states','states','candidate_labels',
                'feasible_labels','deadlock_labels','structural_infeasible_labels','numeric_failure_labels',
                'pairs','duplicate_states_skipped','uninformative_states','perturbed_states',
                'native_evaluations','candidate_native_evaluations'):
        total[key] += detail[key]
    for key in ('state_sources','skipped_state_statuses','teacher_solutions_by_method',
                'skipped_teachers','teacher_state_statuses'):
        for name,value in detail[key].items():
            total[key][name] += value


def build_supervision(*, seed=20261126, states_per_source=2, settings=BRConfig(),
                      dataset='original', normalization=None, layouts=None, progress=print,
                      tasksets=None, label_workers=1):
    """Generate TRAIN labels per layout, then restore the input layout order."""
    if label_workers < 1 or (sys.platform == 'win32' and label_workers > 61):
        raise ValueError('label workers must be positive and at most 61 on Windows')
    normalization = normalization or fit_training_normalization(dataset)
    tasksets = list(selected_training_tasksets(dataset,layouts) if tasksets is None else tasksets)
    names = [name for name,_ in tasksets]
    if len(names) != len(set(names)):
        raise ValueError('duplicate TRAIN layout in supervision input')
    if layouts is not None and (len(layouts) != len(set(layouts)) or set(names) != set(layouts)):
        raise ValueError('supervision tasksets differ from selected TRAIN layouts')
    teachers = read_training_teachers(layouts)
    seeds = {name:seed+i for i,name in enumerate(sorted(names))}
    jobs = [(name,taskset,seeds[name],states_per_source,settings,dataset,normalization,
             {(name,method):teachers[name,method] for method in ('wag_vns','hga') if (name,method) in teachers})
            for name,taskset in tasksets]
    stats = _supervision_stats(dataset,layouts,settings)
    workers = min(label_workers,len(jobs)) if jobs else 0
    stats.update(label_workers_requested=label_workers,label_workers=workers,
                 label_seed_rule='training_seed + zero-based index in sorted selected TRAIN layout IDs',
                 label_training_seed=seed,offline_initialization_uses_attempt_budget=True)
    started = time.perf_counter()
    collected,worker_pids,worker_threads = {},set(),set()
    def announce(*args):
        if progress is not None:
            progress(*args)
    announce('BR_LABELS_START',dict(layouts=len(names),workers=workers,threads_per_worker=1))
    def collect(result):
        name,packed,detail = result
        if name not in seeds or name in collected:
            raise RuntimeError(f'unexpected/duplicate label result: {name}')
        collected[name] = _unpack_label_examples(packed)
        _merge_label_stats(stats,detail)
        worker_pids.add(detail['worker_pid'])
        worker_threads.add((detail['worker_numba_threads'],detail['worker_torch_threads']))
        announce('BR_LABELS',name,dict(completed_layouts=len(collected),total_layouts=len(names),
                 **{k:stats[k] for k in ('states','candidate_labels','pairs')}),
                 'elapsed_s',round(time.perf_counter()-started,3))
    if workers == 1:
        previous_numba = get_num_threads()
        set_num_threads(1)
        try:
            with _torch_threads(1):
                for job in jobs:
                    collect(_label_layout_job(job))
        finally:
            set_num_threads(previous_numba)
    elif workers:
        pool = ProcessPoolExecutor(max_workers=workers,initializer=_initialize_label_worker)
        active = {}
        remaining = iter(jobs)
        try:
            for _ in range(min(len(jobs),2*workers)):
                job = next(remaining)
                active[pool.submit(_label_layout_job,job)] = job[0]
            while active:
                done,_ = wait(active,timeout=10.,return_when=FIRST_COMPLETED)
                if not done:
                    announce('BR_LABELS_WAIT',dict(completed_layouts=len(collected),total_layouts=len(names),
                             elapsed_s=round(time.perf_counter()-started,3)))
                for future in done:
                    name = active.pop(future)
                    try:
                        collect(future.result())
                    except Exception as error:
                        raise RuntimeError(f'TRAIN label generation failed for {name}: {error}') from error
                    job = next(remaining,None)
                    if job is not None:
                        active[pool.submit(_label_layout_job,job)] = job[0]
        finally:
            for future in active:
                future.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
    examples = [example for name in names for example in collected[name]]
    stats.update(layouts=names,layout_records=len(names),unique_layouts=len(set(names)),
                 label_generation_s=time.perf_counter()-started,label_worker_pids=sorted(worker_pids),
                 label_worker_threads=[dict(numba=n,torch=t) for n,t in sorted(worker_threads)])
    if not examples:
        raise RuntimeError('no informative TRAIN labels were generated')
    return examples,stats


def train(args):
    path = args.weights or (SCALE_WEIGHTS if args.dataset == 'scale1000' else WEIGHTS)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('model outputs must stay inside DRL')
    if path.exists():
        try:
            load_model(args.dataset,args.variant,path=path)
        except Exception as error:
            raise ValueError(f'existing checkpoint is incomplete or invalid; file kept unchanged: {path}') from error
        print('BR_TRAIN_ALREADY_SAVED', str(path), flush=True)
        return
    label_workers = getattr(args,'label_workers',1)
    training_threads = getattr(args,'train_threads',1)
    if label_workers < 1 or training_threads < 1 or (sys.platform == 'win32' and label_workers > 61):
        raise ValueError('positive training threads and 1..61 Windows label workers are required')
    started = time.perf_counter()
    settings = BRConfig()
    tasksets, coverage = None, None
    if args.dataset == 'scale1000' and args.layouts is None:
        tasksets, originals = scale3000_tasksets()
        coverage = check_training_teachers(tasksets,originals)
    print('BR_TRAIN_CONFIG',json_text(dict(label_workers=label_workers,threads_per_label_worker=1,
          training_threads=training_threads,epochs=args.epochs,training_seed=args.training_seed)),flush=True)
    normalization = fit_training_normalization(args.dataset,tasksets=tasksets)
    examples, data = build_supervision(seed=args.training_seed, settings=settings,
                                      dataset=args.dataset, normalization=normalization, layouts=args.layouts,
                                      tasksets=tasksets,label_workers=label_workers)
    if args.layouts and {name for name, _, _ in examples} != set(args.layouts):
        raise ValueError('some selected TRAIN layouts produced no informative Pairwise examples')
    if tasksets is not None:
        missing = sorted({name for name,_ in tasksets}-{name for name,_,_ in examples})
        if missing:
            raise ValueError(f'TRAIN layouts without informative Pairwise examples: {missing}')
        data['teacher_coverage'] = coverage
    print('BR_LABEL_SUMMARY', json_text(data), flush=True)
    variants = (args.variant,)
    models, learning = {}, {}
    # Label workers have fully exited before the single model uses CPU threads.
    with _torch_threads(training_threads):
        print('BR_TRAINING_START',json_text(dict(examples=len(examples),threads=torch.get_num_threads(),
              epochs=args.epochs)),flush=True)
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
                    shared_supervision=True,label_workers=label_workers,training_threads=training_threads)
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
    # Reuse an already generated TRAIN graph; this does not schedule or search.
    restored, saved = load_model(args.dataset, args.variant, path=path)
    expected = models[args.variant]
    actual = restored.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(expected[k], actual[k]) for k in expected):
        raise RuntimeError('checkpoint parameters were not restored exactly')
    name, graph, _ = examples[0]
    with torch.inference_mode():
        scores = restored(graph)
    if scores.shape != (len(graph.action_features),) or not torch.isfinite(scores).all():
        raise RuntimeError('reloaded checkpoint produced invalid TRAIN scores')
    print('BR_CHECKPOINT_VERIFIED', json_text(dict(path=str(path), bytes=path.stat().st_size,
        layout_id=name, node_features=len(NODE_FEATURES), robot_features=len(ROBOT_FEATURES),
        edge_features=len(EDGE_FEATURES), action_features=len(ACTION_FEATURES),
        parameters_restored=True, actions=len(scores), finite_scores=True)), flush=True)


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
    parser.add_argument('stage', choices=('prepare', 'train', 'teacher-check', 'validation', 'test', 'summary', 'smoke'))
    parser.add_argument('--dataset', choices=('original', 'scale1000'), default='scale1000')
    parser.add_argument('--training-seed', type=int, default=20261126)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--label-workers','--workers',dest='label_workers',type=int,default=56,
                        help='parallel TRAIN label processes; each uses one thread (Windows: 1..61)')
    parser.add_argument('--train-threads',type=int,default=56,help='CPU threads for the single GNN optimizer process')
    parser.add_argument('--variant', choices=MODEL_VARIANTS, default='full_gnn')
    parser.add_argument('--weights', type=Path, help='new atomic-model checkpoint; historical weights are incompatible')
    parser.add_argument('--layouts', nargs='+', help='distinct TRAIN layout IDs; omitted means all TRAIN for training')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--trial', default='main')
    args = parser.parse_args()
    set_num_threads(1); torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'teacher-check':
        if args.dataset != 'scale1000' or args.layouts is not None:
            parser.error('teacher-check validates full scale1000 TRAIN without --layouts')
        check_training_teachers()
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
        layouts = args.layouts or ('train_01', 'train_12', 'train_30')
        if len(layouts) != len(set(layouts)) or any(name not in tasksets for name in layouts):
            raise ValueError('smoke layouts must be distinct existing TRAIN layouts')
        existing = read_results()
        for name in layouts:
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
    try:
        main()
    except KeyboardInterrupt:
        print('BR_TRAIN_INTERRUPTED: in-memory progress was discarded; saved teachers/checkpoints are kept.',flush=True)
        raise SystemExit(130)
