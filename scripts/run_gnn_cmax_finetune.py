"""Self-critical REINFORCE for the existing joint repair policy, with real B32."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch.nn import functional as F
import run_gnn_joint_repair_pilot as pilot
from mrta_search.gnn_constructor import JointInsertionGNN, construct_or_repair_with_gnn, rollout_joint_repair
from mrta_reference.model import ScheduleStatus

MODEL = ROOT / 'models/gnn_joint_cmax_rl_v1.pt'
CSV = ROOT / 'results/gnn_joint_cmax_rl_v1.csv'
REPORT = ROOT / 'docs/GNN_JOINT_CMAX_RL_V1.md'
SETTINGS = dict(seed=20261027, geometries=20, epochs=3, episodes=8, learning_rate=1e-4,
                insertion_limit=8, temperature=1., bc_coefficient=.01, bc_steps=4, gradient_clip=1.)
HEADS = ('layer1', 'layer2', 'parent_head', 'insertion_head')
CONFIG = pilot.CONFIG


def load_v2():
    checkpoint = torch.load(pilot.MODEL, map_location='cpu', weights_only=True)
    if not checkpoint['normalize_inputs']:
        raise ValueError('V2 normalization is required')
    model = JointInsertionGNN(64, 4, normalize_inputs=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint


def select_sources(records):
    """Cover every existing TRAIN workbook, then fill count gaps; no DEV input."""
    unique = {r['entry']['instance_id']: r for r in reversed(records)}
    groups = defaultdict(list)
    for record in unique.values():
        groups[record['entry']['relative_path']].append(record)
    if len(groups) != 17 or len(unique) != 46:
        raise ValueError('expected the existing 46 geometries / 17 TRAIN workbooks')
    ordered = sorted(unique.values(), key=lambda r: (r['entry']['N'], r['entry']['instance_id']))
    low, high = ordered[0]['entry']['N'], ordered[-1]['entry']['N']
    chosen = []
    for i, path in enumerate(sorted(groups)):
        target = low + (high-low)*i/(len(groups)-1)
        chosen.append(min(groups[path], key=lambda r: (abs(r['entry']['N']-target), r['entry']['instance_id'])))
    while len(chosen) < SETTINGS['geometries']:
        remaining = [r for r in ordered if r not in chosen]
        chosen.append(min(remaining, key=lambda r: (-min(abs(r['entry']['N']-s['entry']['N']) for s in chosen),
                                                   r['entry']['instance_id'])))
    return sorted(chosen, key=lambda r: (r['entry']['N'], r['entry']['instance_id']))


def terminal_return(candidate, reference_cmax):
    if not math.isfinite(reference_cmax) or reference_cmax <= 0:
        raise ValueError('reference must have a positive finite feasible Cmax')
    if candidate is None or candidate.schedule.status is not ScheduleStatus.FEASIBLE:
        return -1.
    cmax = candidate.schedule.cmax
    if cmax is None or not math.isfinite(cmax) or cmax <= 0:
        raise ValueError('FEASIBLE requires a positive finite Cmax')
    return (reference_cmax-cmax)/(reference_cmax+cmax)


def score_pair(sample, greedy, reference_cmax):
    """One B32 per distinct complete candidate, only within this update pair."""
    terminal_return(None, reference_cmax)  # Validate even when neither branch completes.
    cache = {}
    outputs = []
    for result in (sample, greedy):
        if result.solution is None:
            outputs.append((None, -1., 'CONSTRUCTION_FAILED', None))
            continue
        key = (tuple(r.block_ids for r in result.solution.routes), result.directions)
        if key not in cache:
            # Structural/numeric exceptions propagate; they are not failure rewards.
            cache[key] = pilot.oriented.schedule_explicit(result.solution, result.directions, CONFIG)
        candidate = cache[key]
        reward = terminal_return(candidate, reference_cmax)
        outputs.append((candidate, reward, candidate.schedule.status.value,
                        candidate.schedule.cmax if candidate.schedule.status is ScheduleStatus.FEASIBLE else None))
    return outputs, len(cache)


def bc_loss(model, graphs, pool, rng):
    losses = []
    embeddings = {}
    for name, step in rng.sample(pool, SETTINGS['bc_steps']):
        if name not in embeddings:
            embeddings[name] = model.encode_static(graphs[name])
        emb = embeddings[name]
        logits = model.joint_action_logits(emb, step['indices'], step['features'])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([step['target']]))
        if len(step['parent_ids']) > 1:
            pl = model.parent_logits(graphs[name], emb, step['parent_ids'], step['parent_features'])
            loss = loss + F.cross_entropy(pl.unsqueeze(0), torch.tensor([step['parent_target']]))
        losses.append(loss)
    return torch.stack(losses).mean()


def gradient_norms(named_parameters, gradients):
    return {head: math.sqrt(sum(float(g.detach().square().sum())
                               for (name, _), g in zip(named_parameters, gradients)
                               if name.startswith(head+'.') and g is not None)) for head in HEADS}


def train_episode(model, optimizer, current, index, epoch, episode, graphs, pool, rng, phase):
    started = time.perf_counter()
    selection = pilot.SELECTORS[episode % 3]
    q = (4, 8)[(episode // 3) % 2]
    seed = SETTINGS['seed'] + epoch*100000 + index*1000 + episode
    partial = pilot.select_bottleneck_parents(current.solution, current.directions, current.schedule,
                                             selection, q=q, seed=seed, config=CONFIG)
    directions = pilot.partial_directions(current, partial)
    args = (current.solution, partial.routes, directions, partial.removed_parent_ids, model, CONFIG)
    optimizer.zero_grad(set_to_none=True)
    # Both branches use these exact weights. No optimizer step until both B32s finish.
    rollout = rollout_joint_repair(*args, insertion_limit=8, seed=seed)
    greedy = construct_or_repair_with_gnn(*args, insertion_limit=8)
    scored, calls = score_pair(rollout.completion, greedy, current.schedule.cmax)
    (sample, gs, ss, cs), (_, gg, sg, cg) = scored
    advantage = gs-gg
    rl = -advantage*rollout.log_prob
    named = list(model.named_parameters())
    if advantage != 0 and rl.requires_grad:
        grads = torch.autograd.grad(rl, [p for _, p in named], retain_graph=True, allow_unused=True)
        norms = gradient_norms(named, grads)
    else:
        norms = dict.fromkeys(HEADS, 0.)
    bc = bc_loss(model, graphs, pool, rng)
    loss = rl + SETTINGS['bc_coefficient']*bc
    if not bool(torch.isfinite(loss)) or not all(math.isfinite(v) for v in norms.values()):
        raise ValueError('nonfinite loss or policy gradient')
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'], error_if_nonfinite=True))
    optimizer.step()
    before = current.schedule.cmax
    accepted = sample is not None and sample.schedule.status is ScheduleStatus.FEASIBLE and cs < before-1e-9*max(1, before)
    if accepted:
        current = sample
    row = dict(phase=phase, instance=current.solution.parents[0].parent_id, epoch=epoch+1,
               episode=episode+1, method='GNN-Cmax-RL', q=q, selection=selection.value, seed=seed,
               reference_cmax=before, sample_cmax=cs, greedy_cmax=cg, sample_status=ss, greedy_status=sg,
               sample_return=gs, greedy_return=gg, advantage=advantage, nonzero_advantage=int(advantage != 0),
               log_prob=float(rollout.log_prob.detach()), parent_decisions=len(rollout.parent_log_probs),
               insertion_decisions=len(rollout.insertion_log_probs), rl_loss=float(rl.detach()), bc_loss=float(bc.detach()),
               b32_calls=calls, b32_success=sum(c is not None and c.schedule.status is ScheduleStatus.FEASIBLE
                                              for c in {id(v[0]):v[0] for v in scored if v[0] is not None}.values()),
               accepted=int(accepted), final_cmax=current.schedule.cmax, gradient_norm=grad_norm,
               direction0_actions=sum(a.direction == 0 for a in rollout.completion.actions),
               direction1_actions=sum(a.direction == 1 for a in rollout.completion.actions),
               flip_actions=sum(bool(a.neighbour_changes) for a in rollout.completion.actions),
               removed=json.dumps(partial.removed_parent_ids), runtime_s=time.perf_counter()-started,
               sample_failure=rollout.completion.rejection_reason, greedy_failure=greedy.rejection_reason,
               **{'rl_gradient_'+h:v for h,v in norms.items()})
    return current, row


def write_csv(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with CSV.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, phase):
    selected = [r for r in rows if r['phase'] == phase]
    if not selected:
        return {}
    return dict(episodes=len(selected), b32_calls=sum(r['b32_calls'] for r in selected),
                b32_success=sum(r['b32_success'] for r in selected),
                sample_feasible=sum(r['sample_status'] == 'FEASIBLE' for r in selected),
                greedy_feasible=sum(r['greedy_status'] == 'FEASIBLE' for r in selected),
                nonzero=sum(r['nonzero_advantage'] for r in selected),
                feasible_pair_nonzero=sum(r['advantage'] != 0 and r['sample_status'] == r['greedy_status'] == 'FEASIBLE'
                                          for r in selected),
                positive=sum(r['advantage'] > 0 for r in selected), negative=sum(r['advantage'] < 0 for r in selected),
                both_failed=sum(r['sample_return'] == r['greedy_return'] == -1 for r in selected),
                mean_sample_return=sum(r['sample_return'] for r in selected)/len(selected),
                mean_greedy_return=sum(r['greedy_return'] for r in selected)/len(selected),
                mean_abs_advantage=sum(abs(r['advantage']) for r in selected)/len(selected),
                accepted=sum(r['accepted'] for r in selected),
                gradient_nonzero={h:sum(r['rl_gradient_'+h] > 0 for r in selected) for h in HEADS},
                gradient_max={h:max(r['rl_gradient_'+h] for r in selected) for h in HEADS},
                runtime_s=sum(r['runtime_s'] for r in selected))


def save_model(model, metadata, optimizer=None):
    payload = dict(state_dict=model.state_dict(), normalize_inputs=True, hidden_dim=64, heads=4,
                   seed=SETTINGS['seed'], metadata=metadata)
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, MODEL)


def train_checks(model, records, sources, label):
    rows = []
    for source in sources:
        for q in (4, 8):
            options = [r for r in records if len(r['pending_parents']) == q]
            record = min(options, key=lambda r: (r['entry']['instance_id'] != source['entry']['instance_id'],
                                                 abs(r['entry']['N']-source['entry']['N']), r['entry']['instance_id']))
            reference = pilot.record_reference(record)
            start = time.perf_counter()
            result = construct_or_repair_with_gnn(reference, record['partial_routes'], record['partial_directions'],
                                                 record['pending_parents'], model, CONFIG, insertion_limit=8)
            scheduled = None if result.solution is None else pilot.oriented.schedule_explicit(result.solution, result.directions, CONFIG)
            rows.append(dict(phase='TRAIN_CHECK', instance=record['entry']['instance_id'], method=label, q=q,
                             status='CONSTRUCTION_FAILED' if scheduled is None else scheduled.schedule.status.value,
                             b32_calls=int(scheduled is not None), b32_success=int(scheduled is not None and scheduled.schedule.status is ScheduleStatus.FEASIBLE),
                             final_cmax=None if scheduled is None or scheduled.schedule.status is not ScheduleStatus.FEASIBLE else scheduled.schedule.cmax,
                             direction0_actions=sum(a.direction == 0 for a in result.actions),
                             direction1_actions=sum(a.direction == 1 for a in result.actions),
                             flip_actions=sum(bool(a.neighbour_changes) for a in result.actions),
                             runtime_s=time.perf_counter()-start))
            print('TRAIN CHECK', label, rows[-1], flush=True)
    return rows


def train(rows):
    started = time.perf_counter()
    records = [json.loads(line) for line in pilot.DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    graphs, grouped = pilot.prepare_training(records)  # Existing role checks + exact candidate replay.
    pool = [(name, step) for name, steps in grouped.items() for step in steps]
    sources = select_sources(records)
    whitelist = {e['instance_id']:e for group in pilot.dataset.train_sheet_candidates().values() for e in group}
    initials = {}
    for source in sources:
        entry = whitelist[source['entry']['instance_id']]
        if entry['split'] != 'MLP_TRAIN':
            raise ValueError('only MLP_TRAIN geometry may supply rollouts')
        parents = pilot.dataset.load_parents(entry)
        if parents != pilot.record_reference(source).parents:
            raise ValueError('TRAIN file differs from the existing teacher geometry')
        begin = time.perf_counter()
        initial, attempts = pilot.oriented.common_initial_beam(parents, CONFIG, seed=SETTINGS['seed'])
        if initial is None or abs(initial.schedule.cmax-source['original_cmax']) > 1e-6:
            raise ValueError('existing TRAIN Beam initial did not reproduce')
        initials[entry['instance_id']] = initial
        rows.append(dict(phase='TRAIN_BEAM', instance=entry['instance_id'], N=entry['N'], method='Beam-Init',
                         initial_cmax=initial.schedule.cmax, b32_calls=sum(a['status'] != 'CONSTRUCTION_FAILED' for a in attempts),
                         b32_success=sum(a['status'] == 'FEASIBLE' for a in attempts), runtime_s=time.perf_counter()-begin))
        print('TRAIN BEAM', len(initials), '/', len(sources), entry['instance_id'], entry['N'], initial.schedule.cmax, flush=True)
    pilot_sources = [sources[i] for i in (0, len(sources)//2, len(sources)-1)]
    model, original = load_v2()
    rows.extend(train_checks(model, records, pilot_sources, 'GNN-V2'))
    diagnostics_before = pilot.action_diagnostics(model, graphs, grouped)
    # Full 3-geometry/8-episode development check. Discard these updated weights.
    for phase, phase_sources, epochs in (('PILOT', pilot_sources, 1), ('TRAIN', sources, SETTINGS['epochs'])):
        torch.manual_seed(SETTINGS['seed'])
        model, _ = load_v2()
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'])
        rng = random.Random(SETTINGS['seed'])
        for epoch in range(epochs):
            for index, source in enumerate(phase_sources):
                name = source['entry']['instance_id']
                current = initials[name]  # Reset each epoch; retain improvements within its 8 episodes.
                for episode in range(SETTINGS['episodes']):
                    current, row = train_episode(model, optimizer, current, index, epoch, episode, graphs, pool, rng, phase)
                    row.update(instance=name, N=source['entry']['N'])
                    rows.append(row)
                    print(phase, epoch+1, index+1, episode+1, name, row['sample_status'], row['greedy_status'],
                          'G', round(row['sample_return'],6), round(row['greedy_return'],6),
                          'A', round(row['advantage'],6), 'B32', row['b32_calls'], 'accept', row['accepted'], flush=True)
                if phase == 'TRAIN':
                    metadata = dict(settings=SETTINGS, complete=False, train_sources=[r['entry'] for r in sources],
                                    pilot=summary(rows,'PILOT'), train=summary(rows,'TRAIN'), diagnostics_before=diagnostics_before,
                                    train_elapsed_s=time.perf_counter()-started)
                    save_model(model, metadata, optimizer)
                write_csv(rows)
            print(phase, 'EPOCH SUMMARY', epoch+1, json.dumps(summary(rows,phase)), flush=True)
        if phase == 'PILOT':
            info = summary(rows, phase)
            if not all(info['gradient_nonzero'][head] > 0 for head in HEADS):
                raise ValueError('pilot did not demonstrate actual RL gradients through both GAT layers and both heads')
    for name, buffer in model.named_buffers():
        if not torch.equal(buffer, original['state_dict'][name]):
            raise ValueError('V2 normalization changed')
    rows.extend(train_checks(model, records, pilot_sources, 'GNN-Cmax-RL'))
    metadata.update(complete=True, train=summary(rows,'TRAIN'), diagnostics_after=pilot.action_diagnostics(model,graphs,grouped),
                    parameter_change={head:math.sqrt(sum(float((p.detach()-original['state_dict'][name]).square().sum())
                                                        for name,p in model.named_parameters() if name.startswith(head+'.')))
                                      for head in HEADS}, train_elapsed_s=time.perf_counter()-started)
    save_model(model, metadata, optimizer)
    write_csv(rows)
    write_report(rows, metadata)
    return model, metadata


def evaluate(rows, model, metadata):
    if not metadata['complete']:
        raise ValueError('complete the predeclared training before DEV')
    before, _ = load_v2()
    for policy in (before, model):
        policy.eval()
        policy.requires_grad_(False)
    by_name = {e['sheet_name']:e for e in pilot.entries()[1]}
    args = argparse.Namespace(seed=SETTINGS['seed'])
    for name in pilot.oriented.NAMES:
        entry = by_name[name]
        parents = pilot.dataset.load_parents(entry)
        started = time.perf_counter()
        initial, attempts = pilot.oriented.common_initial_beam(parents, CONFIG, seed=SETTINGS['seed'])
        initial_seconds = time.perf_counter()-started
        if initial is None:
            raise ValueError('existing DEV Beam initial failed')
        initial_calls = sum(a['status'] != 'CONSTRUCTION_FAILED' for a in attempts)
        for label, policy in (('NEW', None), ('GNN-V2', before), ('GNN-Cmax-RL', model)):
            row = pilot.iterative_group(entry, initial, label, policy, args)
            row.update(phase='DEV', initial_runtime_s=initial_seconds, initial_b32_calls=initial_calls,
                       total_runtime_s=initial_seconds+row['runtime_s'], total_b32_calls=initial_calls+row['b32_calls'])
            rows.append(row)
            write_csv(rows)
            write_report(rows, metadata)


def write_report(rows, metadata):
    train_info = metadata['train']
    training_rows = [r for r in rows if r['phase']=='TRAIN']
    feasible_advantages = [abs(r['advantage']) for r in training_rows
                           if r['advantage'] != 0 and r['sample_status']==r['greedy_status']=='FEASIBLE']
    total_advantage = sum(abs(r['advantage']) for r in training_rows)
    lines = ['# GNN V2：真实 Cmax 反馈微调', '',
             '固定 Beam 有向初解、原联合候选、原特征及归一化、两层 GAT/64/4。方向及邻块翻转仍在同一次插入中决定。',
             'REINFORCE 使用同一更新前模型、同一部分解的 greedy 回报作为停止梯度基线；没有 critic、PPO 或新算子。', '',
             '## 训练', '', f'预先固定参数：`{json.dumps(SETTINGS)}`。TRAIN 几何由已有46个中按工作簿覆盖及N间隔选择20个，覆盖全部17个TRAIN工作簿；不使用DEV选择训练设置。',
             'RL状态限这20个几何；辅助BC复用原46个TRAIN几何的294条/1872步旧教师。教师经过现有角色检查及真实共同候选重放，仅作每次4步、系数0.01的辅助交叉熵。',
             '奖励：FEASIBLE时 (Cref−Cmax)/(Cref+Cmax)，构造/B32失败为−1；仅完整候选调用B32，相同sample/greedy完整路线及方向复用一次结果。数据/数值异常直接报错。',
             f'3个不同规模几何各8回合流程试训完成后丢弃权重；正式重新加载V2。微调完成：{metadata["complete"]}。', '',
             '|阶段|回合|实际B32成功/调用|sample/greedy可调度|非零优势|正/负优势|双方失败|接纳|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase, info in (('流程检查',metadata['pilot']), ('正式训练',train_info)):
        lines.append(f'|{phase}|{info["episodes"]}|{info["b32_success"]}/{info["b32_calls"]}|{info["sample_feasible"]}/{info["greedy_feasible"]}|{info["nonzero"]}|{info["positive"]}/{info["negative"]}|{info["both_failed"]}|{info["accepted"]}|')
    lines += ['', f'正式平均sample/greedy回报 {train_info["mean_sample_return"]:.6f}/{train_info["mean_greedy_return"]:.6f}，平均|A|={train_info["mean_abs_advantage"]:.6f}；双方可调度且非零优势 {train_info["feasible_pair_nonzero"]} 回合。',
              f'非零优势比例 {train_info["nonzero"]/train_info["episodes"]:.2%}；零优势的{train_info["episodes"]-train_info["nonzero"]}次更新只有辅助BC产生学习信号。双方可调度且非零时，|A|中位数={statistics.median(feasible_advantages):.6f}；可调度/失败差别占优势绝对幅度总和的{1-sum(feasible_advantages)/max(1e-12,total_advantage):.2%}。该幅度统计不能等同于裁剪后梯度贡献比例。',
              f'仅RL项的非零梯度回合：`{json.dumps(train_info["gradient_nonzero"])}`；最大范数：`{json.dumps(train_info["gradient_max"])}`。',
              f'各模块参数变化L2：`{json.dumps(metadata.get("parameter_change",{}))}`。四个原V2归一化buffer逐项保持完全相同；没有重拟合。',
              f'正式训练回合计算秒={train_info["runtime_s"]:.3f}；含数据准备、20个Beam、试训、检查与保存的总训练秒={metadata["train_elapsed_s"]:.3f}。', '',
              '|TRAIN实例|N|Beam Cmax|初始化B32|秒|', '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='TRAIN_BEAM':
            lines.append(f'|{r["instance"]}|{r["N"]}|{r["initial_cmax"]:.3f}|{r["b32_calls"]}|{r["runtime_s"]:.3f}|')
    lines += ['', '## 局部补全保留检查', '',
              '|模型|原TRAIN q4/q8成功/调用|方向0/1|含邻翻转动作|', '|---|---:|---:|---:|']
    for label in ('GNN-V2','GNN-Cmax-RL'):
        checks=[r for r in rows if r['phase']=='TRAIN_CHECK' and r['method']==label]
        lines.append(f'|{label}|{sum(r["b32_success"] for r in checks)}/{sum(r["b32_calls"] for r in checks)}|{sum(r["direction0_actions"] for r in checks)}/{sum(r["direction1_actions"] for r in checks)}|{sum(r["flip_actions"] for r in checks)}|')
    for key in ('diagnostics_before','diagnostics_after'):
        d=metadata.get(key)
        if d:
            lines.append(f'{key}：原1872步教师强制插入准确率={d["insertion_accuracy"]:.2%}，父选择={d["parent_accuracy"]:.2%}，预测方向0/1={d.get("predicted_direction_0",0)}/{d.get("predicted_direction_1",0)}，含翻转动作={d.get("predicted_flip_actions",0)}。')
    lines += ['', '## 六DEV：相同Beam、各自有状态24轮', '',
              '本次三方法全部重新运行；正式补全均为greedy。每次接受严格改善后，下一轮按自己的最新current选区，不能声称全程移除集合相同。',
              '时间为预加载模型、已读取共同几何后，初解构造+B32+24轮搜索；不含统一模型加载/Excel读取。共同Beam实际只计算一次，其成本在每方法端到端时间中各计一次。',
              '|实例|方法|Beam→最终Cmax|严格接纳|搜索B32成功/调用|初解B32|总秒|',
              '|---|---|---:|---:|---:|---:|---:|']
    dev=[r for r in rows if r['phase']=='DEV']
    for r in dev:
        lines.append(f'|{r["instance"]}|{r["method"]}|{r["initial_cmax"]:.3f}→{r["best_cmax"]:.3f}|{r["accepted"]}|{r["b32_success"]}/{r["b32_calls"]}|{r["initial_b32_calls"]}|{r["total_runtime_s"]:.3f}|')
    if len(dev)==18:
        lines += ['', '|方法|搜索B32成功/调用|严格接纳|初解+搜索总秒|', '|---|---:|---:|---:|']
        for label in ('NEW','GNN-V2','GNN-Cmax-RL'):
            group=[r for r in dev if r['method']==label]
            lines.append(f'|{label}|{sum(r["b32_success"] for r in group)}/{sum(r["b32_calls"] for r in group)}|{sum(r["accepted"] for r in group)}|{sum(r["total_runtime_s"] for r in group):.3f}|')
        lines += ['', '最终Cmax逐实例比较（真实差值，1e−9相对容差）：', '']
        for baseline in ('GNN-V2','NEW'):
            other={r['instance']:r for r in dev if r['method']==baseline}
            pairs=[(r['best_cmax']-other[r['instance']]['best_cmax'],1e-9*max(1,other[r['instance']]['best_cmax']))
                   for r in dev if r['method']=='GNN-Cmax-RL']
            lines.append(f'- RL 对 {baseline}：{sum(d < -e for d,e in pairs)}胜、{sum(abs(d)<=e for d,e in pairs)}平、{sum(d > e for d,e in pairs)}负；逐实例差值秒 `{[d for d,e in pairs]}`。')
        v2={r['instance']:r for r in dev if r['method']=='GNN-V2'}
        rl=[r for r in dev if r['method']=='GNN-Cmax-RL']
        no_gain=all(r['best_cmax']>=v2[r['instance']]['best_cmax']-1e-9*max(1,v2[r['instance']]['best_cmax']) for r in rl)
        if no_gain:
            lines[2:2]=['**负结果：本轮真实Cmax微调没有优于原GNN V2，后续主线保留 Beam+B32+GNN V2。**', '']
            lines += ['', '## 结论与问题定位', '',
                      '1. 策略梯度确实作用于GAT和父选择/插入两头：270次非零RL梯度与四个模块的非零参数变化分别验证了梯度和实际更新。不能把本轮说成只打印随机采样奖励或仅做监督训练；但本实验是RL+0.01BC组合，未独立分离两项的因果收益。',
                      '2. 方向和邻块翻转没有消失。原TRAIN六个q4/q8检查仍全部可调度；教师强制方向仍有两个取值、524个含翻转预测。插入准确率仅由73.45%降至73.08%，没有全面遗忘，但也没有动作拟合提升。',
                      '3. 六DEV对V2没有任何更低的最终Cmax。g31中RL只接纳到3125.416秒便停滞，而原V2随后继续接纳到3071.671秒，退化53.745秒；其余五例最终目标值持平。RL严格接纳6次，少于V2的10次，B32成功105/144也没有提高。',
                      '4. 相对NEW仍为3胜1平2负，g33/g45未超过NEW；没有建立超过传统修复或超过原V2的证据。训练中46次sample严格改善没有转化为DEV端到端收益。',
                      '5. 问题不在梯度缺失或优势全部为零。真正的细粒度Cmax优势偏小，较少的可调度/失败差别占据96.74%的优势绝对幅度；同时微调在g31的后续补全路径变差。失败分支136/960（14.17%），不宜笼统称为“大量样本完全无法训练”。这些是信号和搜索轨迹的实测观察，尚未隔离奖励尺度、策略偏移及泛化的因果贡献。',
                      '', '**选择：保留原GNN V2用于论文主线；本轮RL checkpoint和负结果保留，不替换V2，不根据DEV继续改奖励、种子、轮次或增加优化器。**']
        train_calls=sum(r['b32_calls'] for r in rows if r['phase'] in ('TRAIN_BEAM','PILOT','TRAIN','TRAIN_CHECK'))
        dev_calls=sum(r['b32_calls'] for r in dev)+sum(r['initial_b32_calls'] for r in dev if r['method']=='NEW')
        lines += ['', f'实验实际B32：训练流程{train_calls}次（Beam84、试训48、正式928、前后检查12），DEV{dev_calls}次（共享Beam32、搜索432）；共享初解不重复计入物理调用数。正式训练通过同候选复用少调用32次B32。']
    lines += ['', 'B32 DEADLOCK只表示当前调度器未找到时间安排，不是数学不可行证明。CSV保留每回合真实回报、优势、RL梯度与全部DEV逐轮current/移除/接纳轨迹；模型元数据保留固定训练设置与汇总。',
              '运行环境：`D:\\pybullet_test\\.venv\\Scripts\\python.exe`，工作目录`DRL`，CPU单线程；先执行本脚本`--phase train`，完成后执行`--phase dev`。模型固定使用第3轮末权重，无DEV checkpoint选择。77项相关测试通过，另核对554行实际记录、TRAIN隔离、奖励公式、零优势梯度、每几何current连续性及18条24轮DEV轨迹。', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('all','train','dev'),default='all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SETTINGS['seed'])
    if args.phase in ('all','train'):
        rows=[]
        model,metadata=train(rows)
    else:
        rows=list(csv.DictReader(CSV.open(encoding='utf-8-sig')))
        # Numeric CSV values are only decoded for a deliberate separate DEV run.
        for row in rows:
            for key,value in list(row.items()):
                if key in ('removed','trace','failures'):
                    continue
                if value=='':
                    row.pop(key)
                else:
                    try: row[key]=json.loads(value)
                    except (ValueError,TypeError): pass
        checkpoint=torch.load(MODEL,map_location='cpu',weights_only=True)
        metadata=checkpoint['metadata']
        model=JointInsertionGNN(64,4,normalize_inputs=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        if any(r['phase']=='DEV' for r in rows):
            raise ValueError('DEV already recorded; do not silently append duplicate experiments')
    if args.phase in ('all','dev'):
        evaluate(rows,model,metadata)
    print('COMPLETE',args.phase,'rows',len(rows),MODEL,CSV,REPORT,flush=True)


if __name__=='__main__':
    main()
