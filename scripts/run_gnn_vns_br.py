"""TRAIN-only supervised GAT learning and independent GNN-VNS-BR experiments."""
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
from mrta_search.gnn_vns_br import (
    ACTION_FEATURES, NODE_FEATURES, BRConfig, BottleneckGAT, FixedWorkspace,
    candidate_actions, local_graph, ranking_pairs, run_gnn_vns_br, train_ranker,
)
from run_fixed_taskset_experiment import (
    CONFIG, FIELDS, LIMIT, REPORT, RESULTS, SEEDS, json_text, load_tasksets,
    read_rows as read_baselines, solution_record,
)

WEIGHTS = ROOT / 'models/gnn_vns_br.pt'
CSV = RESULTS / 'gnn_vns_br.csv'
OUTPUT_FIELDS = ('trial', *FIELDS)
SECTION = '\n## GNN-VNS-BR 监督排序与瓶颈重构\n'


def read_results():
    if not CSV.exists():
        return []
    csv.field_size_limit(LIMIT)
    with CSV.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def append_result(row):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_FIELDS)
    if not CSV.exists():
        writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if (CSV.stat().st_size if CSV.exists() else 0) + len(text.encode('utf-8')) >= LIMIT:
        raise ValueError('GNN-VNS-BR CSV would exceed 100MB')
    with CSV.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()


def build_supervision(*, seed=20261126, states_per_teacher=2, progress=print):
    """Use best completed B32-feasible TRAIN rows from each existing method.

    The second state is one legal neighborhood perturbation. Candidate labels
    are actual native priority schedules. No VALIDATION/TEST labels are read.
    Samples remain in memory; only training counts go into the checkpoint.
    """
    rng = random.Random(seed)
    tasksets = {name: ts for split, name, ts in load_tasksets() if split == 'train'}
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
    settings = replace(BRConfig(), anchors=8, candidates=8)
    examples = []
    stats = dict(source_split='train', teacher_solutions=0, states=0,
                 candidate_labels=0, feasible_labels=0, infeasible_labels=0,
                 pairs=0, duplicate_states_skipped=0, uninformative_states=0,
                 layouts=[], source_methods=['wag_vns', 'hga'])
    started = time.perf_counter()
    for name in sorted(tasksets):
        workspace = FixedWorkspace(tasksets[name], CONFIG)
        identities = set()
        for method in ('wag_vns', 'hga'):
            row = teachers.get((name, method))
            if row is None:
                continue
            current = workspace.restore(json.loads(row['solution']))
            stats['teacher_solutions'] += 1
            for perturbation in range(states_per_teacher):
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
                graph = local_graph(workspace, current, actions)
                examples.append((name, graph, pairs))
                stats['states'] += 1
                stats['pairs'] += len(pairs)
        stats['layouts'].append(name)
        progress('BR_LABELS', name, {k: stats[k] for k in ('states', 'candidate_labels', 'pairs')},
                 'elapsed_s', round(time.perf_counter() - started, 3))
    stats['label_generation_s'] = time.perf_counter() - started
    if not examples:
        raise RuntimeError('no informative TRAIN labels were generated')
    return examples, stats


def train(args):
    if WEIGHTS.exists() and not args.retrain:
        raise FileExistsError('model exists; use --retrain only for a documented TRAIN/VALIDATION correction')
    if any(r['split'] == 'test' for r in read_results()):
        raise ValueError('TEST has started: the evaluated model cannot be retrained')
    started = time.perf_counter()
    examples, data = build_supervision(seed=args.training_seed)
    model, learning = train_ranker(examples, seed=args.training_seed, epochs=args.epochs)
    training = dict(data=data, learning=learning, seed=args.training_seed,
                    total_offline_s=time.perf_counter() - started,
                    training_s=time.perf_counter() - started - data['label_generation_s'],
                    validation_used_for_labels=False, test_used_for_labels=False)
    checkpoint = dict(model=model.state_dict(), training=training,
                      settings=asdict(BRConfig()), node_features=NODE_FEATURES,
                      action_features=ACTION_FEATURES)
    # The one approved weight file contains the model and its actual training
    # record, with no optimizer, duplicate snapshots or sample sidecars.
    payload = io.BytesIO()
    torch.save(checkpoint, payload)
    if payload.tell() >= LIMIT:
        raise ValueError('model would exceed 100MB')
    WEIGHTS.write_bytes(payload.getvalue())
    print('BR_TRAINED', json_text({k: v for k, v in training.items() if k != 'learning'}), flush=True)
    update_report()


def load_model():
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    model = BottleneckGAT()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def run_one(split, name, taskset, seed, *, trial='main', time_limit=None, model=None, checkpoint=None):
    started = time.perf_counter()
    if model is None:
        model, checkpoint = load_model()
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
    diagnostic.update(final_b32_s=time.perf_counter() - formal_started,
                      formal_diagnostics=schedule.diagnostics,
                      native_feasible=state.feasible, selected_epoch=checkpoint['training']['learning']['selected_epoch'],
                      offline_training_s=checkpoint['training']['total_offline_s'],
                      input_adapter='INDEPENDENT_FIXED_GREEDY_GAT_BOTTLENECK_VNS',
                      single_thread=True, cold_jit_included=True)
    row = dict(trial=trial, split=split, layout_id=name, seed=seed, method='gnn_vns_br',
               status=schedule.status.value, cmax_s=schedule.cmax if schedule.feasible else None,
               native_cmax_s=state.value if state.feasible else None,
               runtime_s=time.perf_counter() - started, iterations=diagnostic['rounds'],
               termination_reason=diagnostic['termination_reason'], b32_calls=1,
               parameters=json_text(asdict(settings)), diagnostic=json_text(diagnostic),
               solution=json_text(solution_record(solution, state.bits)))
    print('BR_RESULT', json_text({k: v for k, v in row.items() if k not in ('solution', 'diagnostic')}), flush=True)
    return row


def experiment(args, split):
    if split == 'test':
        comparison = compare('validation', trial=args.trial)
        if not comparison['ready_for_test']:
            raise ValueError('VALIDATION does not meet the requested continuation conditions; TEST is untouched')
    model, checkpoint = load_model()
    existing = read_results()
    records = [(s, n, ts) for s, n, ts in load_tasksets() if s == split]
    for s, name, taskset in records:
        for seed in SEEDS:
            key = args.trial, split, name, str(seed)
            if any((r['trial'], r['split'], r['layout_id'], r['seed']) == key for r in existing):
                continue
            print('BR_START', split, name, seed, flush=True)
            row = run_one(split, name, taskset, seed, trial=args.trial, model=model, checkpoint=checkpoint)
            append_result(row); existing.append({k: str(v) for k, v in row.items()})
            update_report()
    print('BR_COMPARISON', json_text(compare(split, trial=args.trial)), flush=True)


def compare(split, *, trial='main', result_rows=None):
    records = read_results() if result_rows is None else result_rows
    rows = [r for r in records if r['split'] == split and r['trial'] == trial]
    feasible = [float(r['cmax_s']) for r in rows if r['status'] == 'FEASIBLE' and r['cmax_s']]
    result = dict(split=split, trial=trial, runs=len(rows), feasible=len(feasible),
                  cmax_mean_s=mean(feasible) if feasible else None,
                  cmax_median_s=median(feasible) if feasible else None,
                  runtime_median_s=median(float(r['runtime_s']) for r in rows) if rows else None,
                  paired={})
    for method in ('wag_vns', 'hga'):
        baseline = {(r['layout_id'], r['seed']): r for r in read_baselines(method)
                    if r['split'] == split and r['termination_reason'] == 'ITERATION_LIMIT'}
        changes = []
        for row in rows:
            old = baseline.get((row['layout_id'], str(row['seed'])))
            if row['status'] == 'FEASIBLE' and old and old['status'] == 'FEASIBLE':
                changes.append((float(row['cmax_s']) - float(old['cmax_s'])) / float(old['cmax_s']))
        times = [float(r['runtime_s']) for r in baseline.values()]
        result['paired'][method] = dict(pairs=len(changes),
            median_relative_cmax=median(changes) if changes else None,
            baseline_runtime_median_s=median(times) if times else None,
            runtime_ratio=result['runtime_median_s'] / median(times) if rows and times else None)
    wag, hga = result['paired']['wag_vns'], result['paired']['hga']
    # A 2% paired median margin makes 'competitive with HGA' explicit.
    # This comparison controls only the requested one-time TEST continuation.
    result['ready_for_test'] = bool(split == 'validation' and len(rows) == 30 and len(feasible) >= 28
        and wag['median_relative_cmax'] is not None and wag['median_relative_cmax'] < 0
        and hga['median_relative_cmax'] is not None and hga['median_relative_cmax'] <= .02
        and wag['runtime_ratio'] <= 1.5 and hga['runtime_ratio'] < 1.)
    return result


def update_report(*, checkpoint=None, result_rows=None):
    text = REPORT.read_text(encoding='utf-8').split(SECTION)[0].rstrip()
    lines = [SECTION, '\n当前正式路线为固定拆分→四机器人分配→连续原子任务合并→排序及方向→调度评价。',
             'Adapted-WAG+VNS 与 Adapted-HGA 为保留原始结果的独立对照组；GNN-PPO 作为失败历史保留，不再训练。',
             '\nGNN-VNS-BR 独立执行空间/插入贪心初始化，以原生调度完工时间识别瓶颈，',
             '在瓶颈局部合并作业图上使用两层64维/4头GAT与一个标量候选动作头。',
             'M1–M6 同时处理路线和方向；仅Top-K进入实际原生评价。',
             '原子边界由已有FixedTaskset决定，动作仅迁移现有作业并强制合并连续归属原子。',
             'B32 limited-discrepancy 调度政策保持现有FORMAL_SCOPE_V2。',
             '\n监督标签仅来自已完成且B32可行的TRAIN对照方案及少量合法扰动，',
             '采用实际priority_schedule候选Cmax构造pairwise ranking loss，不可行候选排在可行候选之后。',
             'VALIDATION和TEST独立初始化；训练内部按TRAIN排样留出，不读取VALIDATION/TEST标签。']
    if checkpoint is None and WEIGHTS.exists():
        _, checkpoint = load_model()
    if checkpoint is not None:
        training = checkpoint['training']
        lines += ['\n离线训练实测：', '```json', json.dumps({
            'data': training['data'], 'selected_epoch': training['learning']['selected_epoch'],
            'label_generation_s': training['data']['label_generation_s'],
            'training_s': training['training_s'], 'total_offline_s': training['total_offline_s'],
            'initial_trial_offline_s': checkpoint.get('initial_training', {}).get('total_offline_s'),
            'experiment_notes': checkpoint.get('experiment_notes', [])},
            ensure_ascii=False, indent=2), '```']
    records = read_results() if result_rows is None else result_rows
    trials = sorted({r['trial'] for r in records})
    for trial in trials:
        for split in ('validation', 'test'):
            result = compare(split, trial=trial, result_rows=records)
            if result['runs']:
                lines += ['\n实测配对汇总（仅双方B32可行的同排样同种子计算Cmax变化，失败数量单独保留）：',
                          '```json', json.dumps(result, ensure_ascii=False, indent=2), '```']
    main_validation = compare('validation', result_rows=records)
    if main_validation['runs'] == 30 and not main_validation['ready_for_test']:
        lines += ['\n结论：完整VALIDATION未达到可行率与配对质量要求，TEST未运行。',
                  '仅进行了一次针对性修正及TRAIN重训：M1允许迁移合并作业中的已有端部原子，',
                  '对应轮数上限从100调至200；M5/M6无可行扰动时，在同一有限预算内回退到M1/M2。',
                  '首轮结果以initial区分保留，没有重跑WAG/HGA，也没有改变B32或空间约束。']
    lines += ['\n在线耗时包含独立初始化、GNN编码/推理、候选评价和唯一一次最终B32。',
              '本轮批量串行、单线程，复用TRAIN标签生成和单例阶段已编译的内核；',
              '首次冷JIT包含在初始离线标签计时内。单独启动实验入口时，首例运行计时也包含其编译。离线训练时间单独列示。',
              'VALIDATION要求至少28/30可行、配对优于WAG、相对HGA配对中位退化不超过2%、',
              '中位耗时≤WAG的1.5倍且低于HGA；未满足时不执行TEST。',
              '\n复现入口（指定Python，在DRL目录执行）：', '```powershell',
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py smoke --layout train_01",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py validation",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py test",
              '```']
    output = text + '\n' + '\n'.join(lines) + '\n'
    if len(output.encode('utf-8')) >= LIMIT:
        raise ValueError('existing report would exceed 100MB')
    REPORT.write_text(output, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('train', 'smoke', 'validation', 'test', 'summary', 'all'))
    parser.add_argument('--training-seed', type=int, default=20261126)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--layout', default='train_01')
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--time-limit', type=float)
    parser.add_argument('--trial', default='main')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    set_num_threads(1); torch.set_num_threads(1)
    if args.stage in ('train', 'all'):
        train(args)
    if args.stage in ('smoke', 'all'):
        split, name, taskset = next((s, n, ts) for s, n, ts in load_tasksets() if n == args.layout)
        if split != 'train':
            raise ValueError('smoke testing uses TRAIN only')
        run_one(split, name, taskset, args.seed, time_limit=args.time_limit)
    if args.stage in ('validation', 'all'):
        experiment(args, 'validation')
    if args.stage in ('test', 'all'):
        if args.stage != 'all' or compare('validation', trial=args.trial)['ready_for_test']:
            experiment(args, 'test')
    if args.stage in ('summary', 'all'):
        update_report()
        for split in ('validation', 'test'):
            print('BR_SUMMARY', json_text(compare(split, trial=args.trial)), flush=True)


if __name__ == '__main__':
    main()
