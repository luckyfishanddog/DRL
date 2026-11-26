"""Edge-aware GNN-VNS-BR: TRAIN supervision and bounded TRAIN smoke runs."""
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
    ACTION_FEATURES, EDGE_FEATURES, NODE_FEATURES, BRConfig, BottleneckGAT, FixedWorkspace,
    candidate_actions, local_graph, ranking_pairs, run_gnn_vns_br, train_ranker,
)
from run_fixed_taskset_experiment import (
    CONFIG, FIELDS, LIMIT, REPORT, RESULTS, SEEDS, json_text, load_tasksets,
    read_rows as read_baselines, solution_record,
)

WEIGHTS = ROOT / 'models/gnn_vns_br_edgeaware.pt'
CSV = ROOT / 'results/new_dataset_final/gnn_vns_br_edgeaware.csv'
OUTPUT_FIELDS = ('trial', *FIELDS)
SECTION = '\n## GNN-VNS-BR 边特征与搜索改进（TRAIN Smoke）\n'


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


def build_supervision(*, seed=20261126, states_per_source=2, settings=BRConfig(), progress=print):
    """Use TRAIN teachers, independent greedy starts, and one legal perturbation.

    Candidate budgets and graph construction are identical to online search.
    Labels stay in memory; no validation/test state is turned into a label.
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
    examples = []
    stats = dict(source_split='train', teacher_solutions=0, greedy_initializations=0,
                 states=0, candidate_labels=0, feasible_labels=0, infeasible_labels=0,
                 pairs=0, duplicate_states_skipped=0, uninformative_states=0,
                 layouts=[], state_sources={'wag_vns':0, 'hga':0, 'greedy':0},
                 perturbed_states=0, candidate_parameters=asdict(settings))
    started = time.perf_counter()
    for name in sorted(tasksets):
        workspace = FixedWorkspace(tasksets[name], CONFIG)
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
    if not examples:
        raise RuntimeError('no informative TRAIN labels were generated')
    return examples, stats


def train(args):
    if WEIGHTS.exists() and not args.retrain:
        raise FileExistsError('model exists; use --retrain only for a documented TRAIN/VALIDATION correction')
    started = time.perf_counter()
    settings = BRConfig()
    examples, data = build_supervision(seed=args.training_seed, settings=settings)
    model, learning = train_ranker(examples, seed=args.training_seed, epochs=args.epochs)
    training = dict(data=data, learning=learning, seed=args.training_seed,
                    total_offline_s=time.perf_counter() - started,
                    training_s=time.perf_counter() - started - data['label_generation_s'],
                    validation_used_for_labels=False, test_used_for_labels=False)
    checkpoint = dict(model=model.state_dict(), training=training,
                      settings=asdict(settings), node_features=NODE_FEATURES,
                      edge_features=EDGE_FEATURES, action_features=ACTION_FEATURES)
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
    if (tuple(checkpoint.get('node_features', ())) != NODE_FEATURES
            or tuple(checkpoint.get('edge_features', ())) != EDGE_FEATURES):
        raise ValueError('this entry requires the independent 16-node/3-edge-feature model')
    model = BottleneckGAT()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def run_one(split, name, taskset, seed, *, trial='main', time_limit=None, model=None, checkpoint=None):
    if split != 'train':
        raise ValueError('this round runs TRAIN smoke only; VALIDATION and TEST are untouched')
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
                      input_adapter='FIXED_EDGE_GAT_STRATIFIED_BEST_IMPROVEMENT',
                      single_thread=True, cold_jit_included=True)
    row = dict(trial=trial, split=split, layout_id=name, seed=seed, method='gnn_vns_br_edgeaware',
               status=schedule.status.value, cmax_s=schedule.cmax if schedule.feasible else None,
               native_cmax_s=state.value if state.feasible else None,
               runtime_s=time.perf_counter() - started, iterations=diagnostic['rounds'],
               termination_reason=diagnostic['termination_reason'], b32_calls=1,
               parameters=json_text(asdict(settings)), diagnostic=json_text(diagnostic),
               solution=json_text(solution_record(solution, state.bits)))
    print('BR_RESULT', json_text({k: v for k, v in row.items() if k not in ('solution', 'diagnostic')}), flush=True)
    return row








def update_report(*, checkpoint=None, result_rows=None):
    text = REPORT.read_text(encoding='utf-8').split(SECTION)[0].rstrip()
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
    output = text + '\n' + '\n'.join(lines).replace('FENCE',chr(96)*3) + '\n'
    if len(output.encode('utf-8')) >= LIMIT:
        raise ValueError('existing report would exceed 100MB')
    REPORT.write_text(output, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('train', 'smoke', 'summary'))
    parser.add_argument('--training-seed', type=int, default=20261126)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--layouts', nargs='+', default=('train_01','train_12','train_30'))
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--trial', default='main')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    set_num_threads(1); torch.set_num_threads(1)
    if args.stage == 'train':
        train(args)
    elif args.stage == 'smoke':
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
        update_report()
        for row in read_results():
            print('BR_SMOKE_SUMMARY',json_text({k:v for k,v in row.items() if k not in ('solution','diagnostic')}),flush=True)


if __name__ == '__main__':
    main()
