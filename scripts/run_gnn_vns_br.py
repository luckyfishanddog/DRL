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

from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.fixed_taskset import generate_taskset
from mrta_reference.model import ParentWeld
from mrta_search.gnn_vns_br import (
    ACTION_FEATURES, EDGE_FEATURES, MODEL_VARIANTS, NODE_FEATURES, BRConfig, FixedWorkspace,
    candidate_actions, local_graph, make_ranker, ranking_pairs, run_gnn_vns_br, train_ranker,
)
from run_fixed_taskset_experiment import (
    CONFIG, FIELDS, LIMIT, REPORT, RESULTS, SEEDS, json_text, load_tasksets,
    read_rows as read_baselines, solution_record,
)

WEIGHTS = ROOT / 'models/gnn_vns_br_edgeaware.pt'
CSV = ROOT / 'results/new_dataset_final/gnn_vns_br_edgeaware.csv'
SCALE_WEIGHTS = ROOT / 'models/gnn_vns_br_edgeaware_scale1000.pt'
SCALE_CSV = ROOT / 'results/new_dataset_final/gnn_vns_br_edgeaware_scale1000.csv'
SCALE_DATA = ROOT / 'data/weld_layout/scale1000'
OUTPUT_FIELDS = ('trial', *FIELDS)
SCALE_FIELDS = ('dataset', 'variant', *OUTPUT_FIELDS)
SECTION = '\n## GNN-VNS-BR 边特征与搜索改进（TRAIN Smoke）\n'
SCALE_SECTION = '\n## GNN-VNS-BR scale1000训练、消融与正式实验\n'


def iter_tasksets(split, dataset='original'):
    """Stream new layouts through the unchanged fixed splitter; no data copy."""
    if dataset == 'original':
        yield from ((name, ts) for s, name, ts in load_tasksets() if s == split)
        return
    if dataset != 'scale1000':
        raise ValueError(f'unknown dataset: {dataset}')
    with (SCALE_DATA / f'{split}.jsonl').open(encoding='utf-8') as stream:
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
                      dataset='original', progress=print):
    """Use TRAIN teachers, independent greedy starts, and one legal perturbation.

    Candidate budgets and graph construction are identical to online search.
    Labels stay in memory; no validation/test state is turned into a label.
    """
    rng = random.Random(seed)
    tasksets = training_tasksets(dataset)
    teachers = {}
    for method in ('wag_vns', 'hga'):
        for row in read_baselines(method):
            if (row['split'] != 'train' or row['status'] != 'FEASIBLE'
                    or row['termination_reason'] != 'ITERATION_LIMIT'
                    or row['b32_calls'] != '1' or not row['solution']):
                continue
            key = row['layout_id'], method
            old = teachers.get(key)
            if old is None or float(row['cmax_s']) < float(old['cmax_s']):
                teachers[key] = row
    examples = []
    stats = dict(source_split='train', dataset=dataset, teacher_solutions=0, greedy_initializations=0,
                 generated_states=0, states=0, candidate_labels=0, feasible_labels=0, infeasible_labels=0,
                 pairs=0, duplicate_states_skipped=0, uninformative_states=0,
                 layouts=[], state_sources={'wag_vns':0, 'hga':0, 'greedy':0},
                 perturbed_states=0, candidate_parameters=asdict(settings))
    started = time.perf_counter()
    for name, taskset in tasksets:
        workspace = FixedWorkspace(taskset, CONFIG)
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
                actions = candidate_actions(workspace, current, settings, rng)
                if perturbation and actions:
                    action = rng.choice(actions)
                    current = workspace.evaluate(action.routes, action.bits)
                    actions = candidate_actions(workspace, current, settings, rng)
                identity = tuple((tuple(r), tuple(d)) for r, d in zip(current.routes, current.bits))
                if identity in identities:
                    stats['duplicate_states_skipped'] += 1
                    continue
                identities.add(identity)
                stats['generated_states'] += 1
                if not actions:
                    continue
                values = [workspace.evaluate(a.routes, a.bits).value for a in actions]
                pairs = ranking_pairs(values, rng)
                stats['candidate_labels'] += len(values)
                stats['feasible_labels'] += sum(math.isfinite(v) for v in values)
                stats['infeasible_labels'] += sum(not math.isfinite(v) for v in values)
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
    path = SCALE_WEIGHTS if args.dataset == 'scale1000' else WEIGHTS
    if path.exists():
        print('BR_TRAIN_ALREADY_SAVED', str(path), flush=True)
        return
    started = time.perf_counter()
    settings = BRConfig()
    examples, data = build_supervision(seed=args.training_seed, settings=settings, dataset=args.dataset)
    variants = MODEL_VARIANTS if args.dataset == 'scale1000' else ('full_gnn',)
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
                      edge_features=EDGE_FEATURES, action_features=ACTION_FEATURES)
    if args.dataset == 'scale1000':
        checkpoint.update(models=models, learning=learning)
    else:
        checkpoint['model'] = models['full_gnn']
        training['learning'] = learning['full_gnn']
    # The one approved weight file contains the model and its actual training
    # record, with no optimizer, duplicate snapshots or sample sidecars.
    payload = io.BytesIO()
    torch.save(checkpoint, payload)
    if payload.tell() >= LIMIT:
        raise ValueError('model would exceed 100MB')
    path.write_bytes(payload.getvalue())
    print('BR_TRAINED', json_text({k: v for k, v in training.items() if k != 'learning'}), flush=True)
    (update_scale_report if args.dataset == 'scale1000' else update_report)()


def load_model(dataset='original', variant='full_gnn', *, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(SCALE_WEIGHTS if dataset == 'scale1000' else WEIGHTS,
                                 map_location='cpu', weights_only=False)
    if (tuple(checkpoint.get('node_features', ())) != NODE_FEATURES
            or tuple(checkpoint.get('edge_features', ())) != EDGE_FEATURES):
        raise ValueError('this entry requires the independent 16-node/3-edge-feature model')
    model = make_ranker(variant)
    model.load_state_dict(checkpoint['models'][variant] if dataset == 'scale1000' else checkpoint['model'])
    model.eval()
    return model, checkpoint


def run_one(split, name, taskset, seed, *, trial='main', time_limit=None, model=None, checkpoint=None,
            dataset='original', source='original', variant='full_gnn'):
    if dataset == 'original' and split != 'train':
        raise ValueError('this round runs TRAIN smoke only; VALIDATION and TEST are untouched')
    started = time.perf_counter()
    if model is None:
        model, checkpoint = load_model(dataset, variant)
    settings = BRConfig(**checkpoint['settings'])
    if time_limit is not None:
        settings = replace(settings, time_limit=time_limit)
    workspace, state, diagnostic = run_gnn_vns_br(taskset, model, seed=seed, settings=settings, config=CONFIG)
    solution = workspace.solution(state)
    formal_started = time.perf_counter()
    # Exactly one final call. It uses the unchanged, existing B32 policy.
    # Native arrays contain NumPy scalar genes. The formal evaluator's
    # recovery tie-break serializes directions, which requires Python ints.
    directions = {r: tuple(map(int, bits)) for r, bits in enumerate(state.bits)}
    schedule = reference_schedule_formal(solution, CONFIG, scope=FORMAL_SCOPE_V2,
                                        orientations=directions)
    learning = checkpoint['learning'][variant] if dataset == 'scale1000' else checkpoint['training']['learning']
    diagnostic.update(final_b32_s=time.perf_counter() - formal_started,
                      formal_diagnostics=schedule.diagnostics,
                      native_feasible=state.feasible, selected_epoch=learning['selected_epoch'],
                      offline_training_s=checkpoint['training']['total_offline_s'],
                      input_adapter='FIXED_EDGE_GAT_STRATIFIED_BEST_IMPROVEMENT',
                      single_thread=True, cold_jit_included=True)
    method = f'gnn_vns_br_{variant}_scale1000' if dataset == 'scale1000' else 'gnn_vns_br_edgeaware'
    row = dict(trial=trial, split=split, layout_id=name, seed=seed, method=method,
               status=schedule.status.value, cmax_s=schedule.cmax if schedule.feasible else None,
               native_cmax_s=state.value if state.feasible else None,
               runtime_s=time.perf_counter() - started, iterations=diagnostic['rounds'],
               termination_reason=diagnostic['termination_reason'], b32_calls=1,
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
                  b32_feasible_28=summary['feasible'] >= 28,
                  paired_wag_improves=wag['median_relative_cmax'] is not None and wag['median_relative_cmax'] < 0.,
                  paired_hga_at_most_2pct=hga['median_relative_cmax'] is not None and hga['median_relative_cmax'] <= .02,
                  runtime_vs_wag=runtime is not None and wag['baseline_runtime_median_s'] is not None
                                  and runtime <= 1.5 * wag['baseline_runtime_median_s'],
                  runtime_vs_hga=runtime is not None and hga['baseline_runtime_median_s'] is not None
                                  and runtime < hga['baseline_runtime_median_s'])
    return dict(ready_for_test=all(checks.values()), checks=checks, summary=summary, paired=paired)


def run_experiment(split):
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
    variants = MODEL_VARIANTS if split == 'validation' else ('full_gnn',)
    for variant in variants:
        models[variant], checkpoint = load_model('scale1000', variant, checkpoint=checkpoint)
    keys = [(r['dataset'], r['split'], r['layout_id'], int(r['seed']), r['variant'], r['trial']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate saved model/layout/seed result keys')
    completed = {key for key, row in zip(keys, rows) if row['b32_calls'] == '1' and row['solution']}
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
    text = REPORT.read_text(encoding='utf-8').split(SCALE_SECTION)[0].rstrip()
    if checkpoint is None and SCALE_WEIGHTS.exists():
        checkpoint = torch.load(SCALE_WEIGHTS, map_location='cpu', weights_only=False)
    rows = read_results('scale1000') if result_rows is None else result_rows
    lines = [SCALE_SECTION,
        '实际数据核对：scale1000 JSONL为1000/100/100条。TRAIN已包含原始40条完全相同记录；经用户确认，使用1000个不同TRAIN排样（原始40＋新增960），不重复计为1040个。',
        '沿用generate_taskset逐条转换，不重新生成数据、不保存任务集或监督样本副本。原始40个可复用已完成的WAG/HGA教师，新排样只用独立贪心及合法扰动。',
        '标签只生成一次，Full GNN、屏蔽3维边特征的同结构GNN及18维动作MLP共用完全相同的图、候选与Pairwise标签。训练种子20261126、24轮、学习率0.0003，按完整TRAIN排样留出。',
        '三组沿用相同M1–M6、每邻域40候选/Top-3、实际Cmax全局比较、180秒/200轮/3次扰动。节点16维、边3维、动作18维，Full GAT仍为2层64维4头。',
        '不重跑WAG/HGA、PPO或旧23维GNN；旧边特征模型和3条TRAIN Smoke保留。最终方案仅调用一次原B32，失败记录保留。',
        '本轮产物：models/gnn_vns_br_edgeaware_scale1000.pt；results/new_dataset_final/gnn_vns_br_edgeaware_scale1000.csv；单文件限制100MB。', '']
    lines += ['本轮代码与文件：修改scripts/run_gnn_vns_br.py、src/mrta_search/gnn_vns_br.py、tests/test_model_geometry.py、README.md及本报告；仅新增上述一个checkpoint和一个CSV。',
              '本轮14项直接相关检查已通过，覆盖三模型输入隔离、共同TRAIN留出、原始TRAIN前缀复用、配对准入、CSV中断续写及唯一最终B32；未重复执行全部历史测试。']
    for path in (SCALE_WEIGHTS, SCALE_CSV):
        if path.exists():
            lines.append(f'已保存文件大小：{path.relative_to(ROOT).as_posix()}，{path.stat().st_size}字节。')
    if checkpoint is not None:
        training, data = checkpoint['training'], checkpoint['training']['data']
        data_summary = {k:v for k,v in data.items() if k != 'layouts'}
        first_learning = checkpoint['learning']['full_gnn']
        informative = len(first_learning['train_layouts']) + len(first_learning['held_train_layouts'])
        lines += ['实际监督统计：', 'FENCEjson', json.dumps(data_summary, ensure_ascii=False, indent=2), 'FENCE',
                  f'{data["unique_layouts"]}个输入排样中，{informative}个产生有效Pair；其余{data["unique_layouts"]-informative}个未产生有效Pair，未替换排样或重复生成标签。',
                  f'离线总耗时{training["total_offline_s"]:.3f}秒；三个模型训练合计{training["training_s"]:.3f}秒。', '',
                  '|模型|参数量|训练排样/留出排样|选择轮次|该轮训练Loss|留出Loss|留出Pair准确率|训练秒数|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant in MODEL_VARIANTS:
            record = checkpoint['learning'][variant]
            selected = record['history'][record['selected_epoch'] - 1]
            values = (variant, record['parameters'],
                      f'{len(record["train_layouts"])}/{len(record["held_train_layouts"])}',record['selected_epoch'],
                      selected['train_loss'], selected['held_train_loss'], selected['held_train_pair_accuracy'],record['training_s'])
            lines.append('|'+'|'.join(map(str,values))+'|')
    lines += ['', '正式结果（Cmax只汇总B32可行项，运行时间包括全部成功和失败项）：', '',
              '|数据|划分|模型|完成/计划|B32可行|平均Cmax(s)|中位Cmax(s)|中位在线(s)|原生初始改善中位数|改善有效对数|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('validation','test'):
        for source in ('original','scale1000'):
            variants = MODEL_VARIANTS if split == 'validation' and source == 'original' else ('full_gnn',)
            target = (30 if split == 'validation' else 60) if source == 'original' else 100
            for variant in variants:
                summary = summarize_group(rows, source=source, split=split, variant=variant)
                number = lambda value: '' if value is None else f'{value:.6f}'
                improvement = summary['initial_native_improvement_median']
                values = (source, split, variant, f'{summary["runs"]}/{target}',summary['feasible'],
                          number(summary['cmax_mean_s']),number(summary['cmax_median_s']),number(summary['runtime_median_s']),
                          '' if improvement is None else f'{improvement:.2%}',summary['initial_improvement_pairs'])
                lines.append('|'+'|'.join(map(str,values))+'|')
    original_validation = [r for r in rows if r['dataset'] == 'original' and r['split'] == 'validation'
                           and r['trial'] == 'main']
    full = {(r['layout_id'],int(r['seed'])):r for r in original_validation if r['variant'] == 'full_gnn'}
    controls = {}
    for variant in MODEL_VARIANTS[1:]:
        differences = []
        for row in original_validation:
            other = full.get((row['layout_id'],int(row['seed'])))
            if (row['variant'] == variant and other is not None
                    and row['status'] == other['status'] == 'FEASIBLE'):
                differences.append(float(other['cmax_s']) / float(row['cmax_s']) - 1.)
        controls[variant] = dict(pairs=len(differences), full_median_relative_cmax=median(differences) if differences else None,
                                full_wins=sum(value < -1.e-9 for value in differences),
                                ties=sum(abs(value) <= 1.e-9 for value in differences))
    lines += ['', '消融配对（同原始VALIDATION排样/种子，双方B32可行；负值表示Full GNN更好）：',
              'FENCEjson',json.dumps(controls,ensure_ascii=False,indent=2),'FENCE']
    admission = validation_admission(rows)
    lines += ['', '原始VALIDATION准入及配对比较（负相对变化表示GNN更好）：', 'FENCEjson',
              json.dumps(admission, ensure_ascii=False, indent=2), 'FENCE',
              '初始改善使用同一次搜索的原生初始/最终Cmax；初始不可行时不伪造改善比例，B32最终可行率独立报告。在线耗时含初始化、候选、图/GNN推理和唯一最终B32，首例包含冷JIT；JSONL读取、固定任务转换及checkpoint加载在单例计时之外，与原对照任务集预处理口径一致。']
    if admission['summary']['runs'] == 30:
        mismatch = sum(r['status'] != 'FEASIBLE' and json.loads(r['diagnostic'])['native_feasible']
                       for r in full.values())
        lines += ['', f'结果分析：原始Full GNN的原生可行数为{admission["summary"]["native_feasible"]}/30，'
                  f'B32可行数为{admission["summary"]["feasible"]}/30；其中{mismatch}次原生可行却在最终B32失败。'
                  '这说明优化原生Cmax不能直接保证B32可行，优先需要核对现有快速评价与B32调度结果之间的差异。',
                  '现有六邻域确实生成并评价了可行候选，MLP采用完全相同的生成与搜索预算；应结合配对Cmax判断GNN排序收益，不能以内部Pair准确率替代正式搜索效果。']
        mlp_delta = controls['action_mlp']['full_median_relative_cmax']
        if mlp_delta is not None and mlp_delta > 0:
            lines.append(f'Full GNN相对MLP的双可行配对Cmax中位退化为{mlp_delta:.2%}；本轮未证明图编码提供额外的Cmax收益。'
                         '这些结果首先提示排序目标与正式可行性/搜索收益的衔接问题，尚不足以把问题归因于候选覆盖不足；未追加模块、超参数搜索或重训。')
    test_rows = [r for r in rows if r['split'] == 'test']
    if test_rows:
        lines += ['', '原始TEST配对比较：', 'FENCEjson',
                  json.dumps(paired_comparison(rows, 'test'), ensure_ascii=False, indent=2), 'FENCE']
    else:
        lines += ['', 'TEST尚未执行。只有原始30次Full GNN VALIDATION满足既有全部准入标准后才允许执行。'
                  if admission['ready_for_test'] else
                  '原始Full GNN VALIDATION未通过既有准入标准，本轮不运行TEST。']
    for source in ('original','scale1000'):
        for variant in (MODEL_VARIANTS if source == 'original' else ('full_gnn',)):
            summary = summarize_group(rows, source=source, split='validation', variant=variant)
            if summary['runs']:
                lines += ['', '搜索诊断：', 'FENCEjson', json.dumps(summary,ensure_ascii=False,indent=2), 'FENCE']
    failures = [r for r in rows if r['status'] != 'FEASIBLE']
    if failures:
        lines += ['', '失败记录：', '', '|数据|划分|排样|种子|模型|B32状态|原生可行|停止原因|', '|---|---|---|---:|---|---|---|---|']
        for r in failures:
            detail = json.loads(r['diagnostic'])
            values = (r['dataset'],r['split'],r['layout_id'],r['seed'],r['variant'],r['status'],
                      detail['native_feasible'],r['termination_reason'])
            lines.append('|'+'|'.join(map(str,values))+'|')
    lines += ['', '本轮复现入口（本地DRL、指定解释器）：', 'FENCEpowershell',
              *[f"& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py {stage} --dataset scale1000"
                for stage in ('train','validation','test','summary')], 'FENCE',
              'train发现已有本轮checkpoint时直接复用；正式CSV按数据集、划分、排样、种子和模型读取已有完整记录，失败的完整B32记录同样保留且不重跑。']
    output = text + '\n' + '\n'.join(lines).replace('FENCE',chr(96)*3) + '\n'
    if len(output.encode('utf-8')) >= LIMIT:
        raise ValueError('existing report would exceed 100MB')
    REPORT.write_text(output, encoding='utf-8')


def update_readme():
    if not SCALE_WEIGHTS.exists():
        return
    checkpoint = torch.load(SCALE_WEIGHTS, map_location='cpu', weights_only=False)
    data = checkpoint['training']['data']
    rows = read_results('scale1000')
    admission = validation_admission(rows)
    lines = ['\n## scale1000实测结果\n',
             f'实际输入{data["unique_layouts"]}个不同TRAIN排样，生成{data["generated_states"]}个状态，'
             f'其中{data["states"]}个有效状态、{data["candidate_labels"]}个候选标签、{data["pairs"]}对排序样本。'
             f'三个模型共用一次标签，均完成24轮；离线总耗时{checkpoint["training"]["total_offline_s"]:.2f}秒。', '',
             '|原始VALIDATION模型|B32可行/运行数|平均Cmax(s)|中位Cmax(s)|中位在线(s)|',
             '|---|---:|---:|---:|---:|']
    number = lambda value: '' if value is None else f'{value:.2f}'
    for variant in MODEL_VARIANTS:
        group = summarize_group(rows, source='original', split='validation', variant=variant)
        values = (variant, f'{group["feasible"]}/{group["runs"]}',number(group['cmax_mean_s']),
                  number(group['cmax_median_s']),number(group['runtime_median_s']))
        lines.append('|'+'|'.join(map(str,values))+'|')
    fresh = summarize_group(rows, source='scale1000', split='validation', variant='full_gnn')
    lines += ['', f'新VALIDATION：Full GNN的B32可行数为{fresh["feasible"]}/{fresh["runs"]}，'
              f'平均/中位Cmax为{number(fresh["cmax_mean_s"])}/{number(fresh["cmax_median_s"])}秒，'
              f'中位在线耗时{number(fresh["runtime_median_s"])}秒。',
              'Cmax仅汇总B32可行结果，运行时间包含失败运行；全部记录保存在新CSV。',
              f'原始VALIDATION准入：{"通过" if admission["ready_for_test"] else "未通过"}；'
              f'已保存TEST记录{sum(r["split"] == "test" for r in rows)}条。'
              '配对差异、初始改善、失败详情及训练记录见[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。']
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8').split('\n## scale1000实测结果\n')[0].rstrip()
    output = text + '\n' + '\n'.join(lines) + '\n'
    if len(output.encode('utf-8')) >= LIMIT:
        raise ValueError('README would exceed 100MB')
    path.write_text(output, encoding='utf-8')


def update_report(*, checkpoint=None, result_rows=None):
    previous = REPORT.read_text(encoding='utf-8')
    scale_tail = SCALE_SECTION + previous.split(SCALE_SECTION, 1)[1] if SCALE_SECTION in previous else ''
    text = previous.split(SECTION)[0].rstrip()
    if checkpoint is None and WEIGHTS.exists():
        _, checkpoint = load_model()
    rows = read_results() if result_rows is None else result_rows
    lines = [SECTION,
        '本轮仅使用TRAIN监督训练和约3个TRAIN排样单种子Smoke；未运行完整VALIDATION、TEST或WAG/HGA对照。',
        '旧模型models/gnn_vns_br.pt及固定任务目录的gnn_vns_br.csv已从原进程内存保存，60条结果未重新计算。',
        '旧权重与旧实验结果不覆盖；本轮权重和CSV分别位于models/gnn_vns_br_edgeaware.pt及results/new_dataset_final/gnn_vns_br_edgeaware.csv。',
        '',
        '节点输入16维：类型2维，作业端点4维、长度/位置/方向3维，机器人身份4维、负载/完工/空走3维。',
        '作业节点只填作业特征，机器人状态仅放在机器人节点。',
        '边输入3维：can_execute、is_current_owner、directed_empty_time。每个作业关联全部4个机器人，资格直接读取w.eligible。',
        '后继、前驱和空间转移各按当前方向读取w.travel，再除以workspace.scale，与机器人时间状态使用同一归一化尺度；反向成本独立计算。空间邻居按有向转移时间稀疏选择。',
        '连续边特征通过每层一个3→4线性映射直接进入注意力logits；仍为两层64维/4头共享GAT和单个动作评分头，18维动作特征保持。',
        '',
        '训练与推理共用BRConfig、candidate_actions和local_graph，alpha=8、anchors=12、每邻域最多40候选。',
        'TRAIN来源包括既有WAG/HGA的B32可行完整方案、独立贪心初始化及少量合法扰动；按完整TRAIN排样留出，不使用VALIDATION/TEST标签。',
        '候选按邻域、目标机器人、锚点轮转分配有限预算，先用真实资格排除非法转移；取消随机截断及路线代价的半数预淘汰。',
        '每个邻域最多3个高分候选接受原生评价，在全部六邻域中取实际Cmax最小的可行改善。',
        '有限M5/M6扰动及M1/M2回退保留全局可行最优；180秒上限、200轮和3次扰动预算保持。',
        '',
        '清理：删除3个无调用方旧脚本；删除候选执行/生成中的拆分激活、取消、点切换分支及活动算子注册。',
        '旧主CLI移除53个动态GNN阶段；CEM/JN-MIP/RCG和MODE_NEW仅保留结果摘要/审计入口，旧PPO搜索CLI移除。',
        '另移除cmax_finetune、joint_repair_pilot、joint_oriented_repair_pilot及eg_gnn_pvnd的动态搜索主入口，保留其历史读取及共享辅助函数。',
        '保留历史枚举、几何校验和结果读取，以及WAG/HGA共享函数；FORMAL_SCOPE_V2、B32与对照算法主体没有修改。',
    ]
    if checkpoint is not None:
        training = checkpoint['training']
        lines += ['', '离线训练记录：', 'FENCEjson',
            json.dumps(dict(data=training['data'], selected_epoch=training['learning']['selected_epoch'],
                            training_s=training['training_s'], total_offline_s=training['total_offline_s']),
                       ensure_ascii=False, indent=2), 'FENCE']
    if rows:
        lines += ['', '|排样|种子|原生可行|B32状态|B32 Cmax(s)|在线耗时(s)|',
                      '|---|---:|---|---|---:|---:|']
        for row in rows:
            detail = json.loads(row['diagnostic'])
            lines.append('|'+ '|'.join(map(str, (row['layout_id'],row['seed'],detail['native_feasible'],
                                                row['status'],row['cmax_s'],row['runtime_s'])))+'|')
        feasible = [float(r['cmax_s']) for r in rows if r['status']=='FEASIBLE' and r['cmax_s']]
        lines += ['', f'Smoke原生可行{sum(json.loads(r["diagnostic"])["native_feasible"] for r in rows)}/{len(rows)}，'
                      f'B32可行{len(feasible)}/{len(rows)}；这不是完整VALIDATION，也不据此宣称总体优于对照。']
    lines += ['', '在线耗时包含独立初始化、图构造/GNN推理、候选评价及唯一一次最终B32。独立启动Smoke的首例含冷JIT，后两例复用进程内编译。',
              '', '本轮复现入口（在本地DRL目录）：', 'FENCEpowershell',
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py smoke",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py summary", 'FENCE']
    output = text + '\n' + '\n'.join(lines).replace('FENCE',chr(96)*3) + '\n' + scale_tail
    if len(output.encode('utf-8')) >= LIMIT:
        raise ValueError('existing report would exceed 100MB')
    REPORT.write_text(output, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('train', 'validation', 'test', 'summary', 'smoke'))
    parser.add_argument('--dataset', choices=('original', 'scale1000'), default='scale1000')
    parser.add_argument('--training-seed', type=int, default=20261126)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--layouts', nargs='+', default=('train_01','train_12','train_30'))
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--trial', default='main')
    args = parser.parse_args()
    set_num_threads(1); torch.set_num_threads(1)
    if args.stage == 'train':
        train(args)
    elif args.stage in ('validation', 'test'):
        if args.dataset != 'scale1000':
            raise ValueError('formal stages use the new scale1000 checkpoint; old experiments are preserved')
        run_experiment(args.stage)
    elif args.stage == 'smoke':
        if args.dataset != 'original':
            raise ValueError('the previous TRAIN smoke uses --dataset original; this round uses validation')
        model, checkpoint = load_model()
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
