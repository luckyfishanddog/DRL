"""Train the fixed-split assignment GNN and run the first RULE/GNN VNS pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from mrta_baselines.common import BaselineStatus, CommonBaselineEvaluator
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns_v2
from mrta_data import phase3_split
from mrta_ranker import dataset as ranker_dataset
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map
from mrta_search.gnn_constructor import (
    BlockAssignmentGNN,
    assignment_targets,
    build_block_robot_graph,
    construct_solution,
    dynamic_features,
    fixed_blocks,
    fixed_split_patterns,
    rule_assignment,
)
from mrta_search.vns import (
    GLOBAL_BALANCED,
    LEGACY_LOCAL_ABSOLUTE,
    NEIGHBORHOODS,
    VNSResult,
    VNSStats,
    evaluate_initial_with_shared_repair,
    run_vns,
)


MODEL_PATH = ROOT / "models" / "gnn_constructor_v1.pt"
REPORT_PATH = ROOT / "docs" / "GNN_CONSTRUCT_VNS_PILOT.md"
TRAINING_SEED = 20261022
PILOT_SEEDS = (20261022, 20261023)


def _mean(values):
    rows = [value for value in values if value is not None]
    return statistics.fmean(rows) if rows else None


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _entry_label(entry) -> str:
    return f"{Path(entry['relative_path']).stem}:{entry['sheet_name']} (N={entry['N']})"


def _teacher_loss(
    model: BlockAssignmentGNN,
    parents,
    expert_solution,
    config: ScientificConfig,
) -> tuple[torch.Tensor, int, int]:
    patterns = fixed_split_patterns(parents, config)
    blocks = fixed_blocks(parents, patterns, config)
    graph = build_block_robot_graph(blocks, patterns, config)
    embeddings = model.encode_static(graph)
    targets = assignment_targets(expert_solution)
    assigned: list[list[str]] = [[] for _ in range(4)]
    by_id = {block.block_id: block for block in blocks}
    losses = []
    correct = 0
    legal = 0
    for step, block in enumerate(blocks):
        target = targets[block.block_id]
        mask = torch.tensor(
            [robot_is_eligible(block, robot, config) for robot in range(4)],
            dtype=torch.bool,
        )
        if not bool(mask[target]):
            raise ValueError(f"expert has illegal target {block.block_id} -> R{target}")
        logits = model.action_logits(
            graph,
            embeddings,
            block.block_id,
            dynamic_features(block, assigned, by_id, config, step, len(blocks)),
            mask,
        )
        losses.append(nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
        predicted = int(torch.argmax(logits).item())
        correct += int(predicted == target)
        legal += int(bool(mask[predicted]))
        assigned[target].append(block.block_id)
    return torch.stack(losses).mean(), correct, legal


def _evaluate_model(model, records, config):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    legal = 0
    coverage = 0
    with torch.inference_mode():
        for record in records:
            loss, hit, valid = _teacher_loss(
                model, record["parents"], record["expert"], config
            )
            count = sum(len(route.block_ids) for route in record["expert"].routes)
            total_loss += float(loss) * count
            total_steps += count
            correct += hit
            legal += valid
            constructed = construct_solution(record["parents"], config, model=model)
            expected = set(block_map(record["expert"], config))
            actual = {
                item for route in constructed.solution.routes for item in route.block_ids
            }
            coverage += int(actual == expected and sum(map(len, (route.block_ids for route in constructed.solution.routes))) == len(expected))
    return {
        "loss": total_loss / max(1, total_steps),
        "accuracy": correct / max(1, total_steps),
        "legal_action_fraction": legal / max(1, total_steps),
        "coverage_fraction": coverage / max(1, len(records)),
        "steps": total_steps,
    }


def _expert_record(entry, config, budget, seed):
    parents = ranker_dataset.load_parents(entry)
    construction = construct_solution(parents, config)
    evaluator = CommonBaselineEvaluator(
        parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=budget,
        checkpoints=(budget,),
    )
    result = run_vns(
        construction.solution,
        evaluator,
        config,
        seed=seed,
        per_neighborhood_budget=10,
    )
    if result.initial is None or result.best is None:
        return None, {
            "entry": entry,
            "status": "NO_CERTIFIED_EXPERT",
            "initial_failures": result.initialization_failures,
        }
    return {
        "entry": entry,
        "parents": parents,
        "expert": result.best.solution,
    }, {
        "entry": entry,
        "status": "CERTIFIED",
        "initial_cmax": result.initial.metrics.cmax,
        "final_cmax": result.best.metrics.cmax,
        "iterations": result.stats.iterations,
        "improvements": dict(result.stats.improvements),
    }


def build_experts(entries, config, budget):
    records = []
    diagnostics = []
    for index, entry in enumerate(entries):
        print(
            f"expert {index + 1}/{len(entries)} {_entry_label(entry)}",
            flush=True,
        )
        record, diagnostic = _expert_record(
            entry, config, budget, TRAINING_SEED + index
        )
        diagnostics.append(diagnostic)
        if record is not None:
            records.append(record)
    return records, diagnostics


def train_model(train_records, dev_records, config, epochs):
    random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = BlockAssignmentGNN(hidden_dim=64, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = None
    best_epoch = 0
    best_dev = math.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_records)))
        random.Random(TRAINING_SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in order:
            record = train_records[index]
            loss, _, _ = _teacher_loss(
                model, record["parents"], record["expert"], config
            )
            losses.append(loss)
        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_metrics = _evaluate_model(model, train_records, config)
        dev_metrics = _evaluate_model(model, dev_records, config)
        history.append(
            {
                "epoch": epoch,
                "optimization_loss": float(train_loss.detach()),
                "train": train_metrics,
                "dev": dev_metrics,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} dev_acc={dev_metrics['accuracy']:.3f}",
            flush=True,
        )
        if dev_metrics["loss"] < best_dev - 1.0e-12:
            best_dev = dev_metrics["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    final_train = _evaluate_model(model, train_records, config)
    final_dev = _evaluate_model(model, dev_records, config)
    training_seconds = time.perf_counter() - started
    checkpoint = {
        "model_type": "BLOCK_ASSIGNMENT_GNN_V1",
        "hidden_dim": 64,
        "heads": 4,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "training_seed": TRAINING_SEED,
        "train_metrics": final_train,
        "dev_metrics": final_dev,
        "training_seconds": training_seconds,
    }
    torch.save(checkpoint, MODEL_PATH)
    return model, {
        **checkpoint,
        "history": history,
        "state_dict": None,
    }


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "BLOCK_ASSIGNMENT_GNN_V1":
        raise ValueError("not a block-assignment GNN checkpoint")
    model = BlockAssignmentGNN(checkpoint["hidden_dim"], checkpoint["heads"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _solution_metrics(candidate, config):
    if candidate is None or candidate.metrics is None or candidate.schedule is None:
        return {
            "cmax": None,
            "counts": None,
            "loads": None,
            "empty": None,
            "waiting": None,
        }
    blocks = block_map(candidate.solution, config, scope=FORMAL_SCOPE_V2)
    counts = [len(route.block_ids) for route in candidate.solution.routes]
    loads = [
        sum(config.process_time(blocks[item].length) for item in route.block_ids)
        for route in candidate.solution.routes
    ]
    return {
        "cmax": candidate.metrics.cmax,
        "counts": counts,
        "loads": loads,
        "empty": candidate.metrics.total_empty_travel,
        "waiting": candidate.metrics.total_waiting,
    }


def _assignment_diagnostics(solution, config):
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    shared = []
    route_order = []
    for route in solution.routes:
        mids = [
            (blocks[item].start[0] + blocks[item].end[0]) / 2.0
            for item in route.block_ids
        ]
        ordered_pairs = max(0, len(mids) - 1)
        matching_pairs = sum(
            mids[index + 1] >= mids[index]
            for index in range(ordered_pairs)
        )
        route_order.append(matching_pairs / max(1, ordered_pairs))
        for block_id in route.block_ids:
            block = blocks[block_id]
            if all(robot_is_eligible(block, robot, config) for robot in range(4)):
                shared.append(route.robot_id)
    return {
        "shared_counts": [shared.count(robot) for robot in range(4)],
        "increasing_x_pair_fraction": route_order,
    }


def run_method(entry, method, seed, config, budget):
    parents = ranker_dataset.load_parents(entry)
    started = time.perf_counter()
    model = None
    model_load_seconds = 0.0
    if method.startswith("GNN"):
        load_started = time.perf_counter()
        model, _ = load_model()
        model_load_seconds = time.perf_counter() - load_started
    construction = construct_solution(parents, config, model=model)
    remaining = max(0.01, budget - (time.perf_counter() - started))
    evaluator = CommonBaselineEvaluator(
        parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=remaining,
        checkpoints=(remaining,),
    )
    if method.endswith("+VNS"):
        outcome = run_vns(
            construction.solution,
            evaluator,
            config,
            seed=seed,
            per_neighborhood_budget=12,
        )
    else:
        initial, attempts, failures = evaluate_initial_with_shared_repair(
            construction.solution, evaluator, config
        )
        outcome = VNSResult(
            initial,
            initial,
            VNSStats(),
            attempts,
            failures,
            evaluator.elapsed,
        )
    elapsed = time.perf_counter() - started
    initial_metrics = _solution_metrics(outcome.initial, config)
    final_metrics = _solution_metrics(outcome.best, config)
    solution = outcome.best.solution if outcome.best is not None else construction.solution
    return {
        "method": method,
        "seed": seed,
        "entry": entry,
        "status": "CERTIFIED" if outcome.best is not None else "INITIALIZATION_FAILED",
        "initial": initial_metrics,
        "final": final_metrics,
        "solve_seconds": elapsed,
        "model_load_seconds": model_load_seconds,
        "gnn_inference_seconds": construction.inference_seconds if model is not None else 0.0,
        "legal_action_fraction": construction.legal_action_fraction,
        "iterations": outcome.stats.iterations,
        "attempted": dict(outcome.stats.attempted),
        "improvements": dict(outcome.stats.improvements),
        "failures": {
            **dict(outcome.stats.failures),
            "INITIAL": list(outcome.initialization_failures),
        },
        "assignment": _assignment_diagnostics(solution, config),
    }


def _solution_changes(initial, final):
    initial_owner = {
        block_id: route.robot_id
        for route in initial.solution.routes
        for block_id in route.block_ids
    }
    final_owner = {
        block_id: route.robot_id
        for route in final.solution.routes
        for block_id in route.block_ids
    }
    assignment_changes = sum(
        initial_owner[block_id] != final_owner[block_id] for block_id in initial_owner
    )
    changed_routes = sum(
        left.block_ids != right.block_ids
        for left, right in zip(initial.solution.routes, final.solution.routes)
    )
    return assignment_changes, changed_routes


def run_paired_vns_fix(entry, construction_kind, seed, config, budget, model=None):
    """Compare legacy/new selection from one already-certified fixed-split initial."""
    parents = ranker_dataset.load_parents(entry)
    constructed = construct_solution(
        parents,
        config,
        model=model if construction_kind == "GNN" else None,
    )
    initial_evaluator = CommonBaselineEvaluator(
        parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=max(5.0, budget),
        checkpoints=(max(5.0, budget),),
    )
    initial, attempts, initial_failures = evaluate_initial_with_shared_repair(
        constructed.solution, initial_evaluator, config
    )
    if initial is None:
        return [{
            "instance": entry["instance_id"],
            "N": entry["N"],
            "construction": construction_kind,
            "seed": seed,
            "status": "INITIALIZATION_FAILED",
            "initial_attempts": attempts,
            "initial_failures": initial_failures,
        }]

    rows = []
    for policy in (LEGACY_LOCAL_ABSOLUTE, GLOBAL_BALANCED):
        evaluator = CommonBaselineEvaluator(
            parents,
            config,
            FORMAL_SCOPE_V2,
            time_limit=budget,
            checkpoints=(budget,),
        )
        outcome = run_vns(
            initial.solution,
            evaluator,
            config,
            seed=seed,
            per_neighborhood_budget=12,
            selection_policy=policy,
            certified_initial=initial,
        )
        assert outcome.initial is not None and outcome.best is not None
        assignment_changes, changed_routes = _solution_changes(
            outcome.initial, outcome.best
        )
        initial_metrics = _solution_metrics(outcome.initial, config)
        final_metrics = _solution_metrics(outcome.best, config)
        rows.append({
            "instance": entry["instance_id"],
            "N": entry["N"],
            "construction": construction_kind,
            "policy": policy,
            "seed": seed,
            "status": "CERTIFIED",
            "initial": initial_metrics,
            "final": final_metrics,
            "improvement_percent": 100.0 * (
                initial_metrics["cmax"] - final_metrics["cmax"]
            ) / initial_metrics["cmax"],
            "solve_seconds": outcome.elapsed,
            "iterations": outcome.stats.iterations,
            "generated": dict(outcome.stats.generated),
            "attempted": dict(outcome.stats.attempted),
            "certified": dict(outcome.stats.certified),
            "improvements": dict(outcome.stats.improvements),
            "failures": dict(outcome.stats.failures),
            "explored_types": outcome.stats.explored_types,
            "assignment_changes": assignment_changes,
            "changed_routes": changed_routes,
        })
    return rows


def run_vns_fix_mode(dev_entries, config, budget, all_dev, seeds):
    entries = dev_entries if all_dev else _select_pilot_entries(dev_entries)
    model, _ = load_model()
    rows = []
    for entry in entries:
        for seed in seeds:
            for construction_kind in ("RULE", "GNN"):
                print(
                    f"paired-vns {_entry_label(entry)} seed={seed} {construction_kind}",
                    flush=True,
                )
                produced = run_paired_vns_fix(
                    entry,
                    construction_kind,
                    seed,
                    config,
                    budget,
                    model,
                )
                rows.extend(produced)
                for row in produced:
                    print("VNS_FIX_ROW " + json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def run_wag_reference(entry, seed, config, budget):
    parents = ranker_dataset.load_parents(entry)
    started = time.perf_counter()
    result = run_adapted_wag_vns_v2(
        parents,
        config,
        AdaptedWAGConfig(),
        seed=seed,
        time_limit=budget,
        checkpoints=(budget,),
    )
    return {
        "method": "ADAPTED_WAG_VNS_V2_OPTIONAL_SPLITS",
        "seed": seed,
        "entry": entry,
        "status": result.status.value,
        "cmax": None if result.metrics is None else result.metrics.cmax,
        "solve_seconds": time.perf_counter() - started,
    }


def _select_pilot_entries(entries):
    selected = []
    for target in (35, 55, 85):
        candidates = [entry for entry in entries if entry["target_N"] == target]
        selected.append(min(candidates, key=lambda entry: (abs(entry["N"] - target), entry["instance_id"])))
    return selected


def _rule_agreement(records, model, config):
    total = 0
    rule_hits = 0
    gnn_hits = 0
    shared_total = 0
    shared_gnn_hits = 0
    for record in records:
        parents = record["parents"]
        patterns = fixed_split_patterns(parents, config)
        blocks = fixed_blocks(parents, patterns, config)
        targets = assignment_targets(record["expert"])
        rule = dict(rule_assignment(blocks, config))
        gnn = dict(construct_solution(parents, config, model=model).assignments)
        for block in blocks:
            target = targets[block.block_id]
            total += 1
            rule_hits += int(rule[block.block_id] == target)
            gnn_hits += int(gnn[block.block_id] == target)
            if all(robot_is_eligible(block, robot, config) for robot in range(4)):
                shared_total += 1
                shared_gnn_hits += int(gnn[block.block_id] == target)
    return {
        "rule_expert_accuracy": rule_hits / max(1, total),
        "gnn_expert_accuracy": gnn_hits / max(1, total),
        "shared_gnn_expert_accuracy": shared_gnn_hits / max(1, shared_total),
        "shared_steps": shared_total,
    }


def write_report(training, train_diag, dev_diag, pilot, wag_rows, pilot_entries, agreement, args):
    methods = ("RULE", "RULE+VNS", "GNN", "GNN+VNS")
    lines = [
        "# GNN Construct + VNS Pilot",
        "",
        "## 实际实现",
        "",
        "本轮固定拆分搜索空间：父焊缝能整段执行时取 WHOLE；否则从现有合法目录确定性选择第一个 mandatory Y_SPLIT。没有激活可选 Y/X 拆分，结果不能视为完整 V2 可选拆分问题的最优结果。共享覆盖仍只是任务层假设，不代表 URDF 或真实机械臂逆运动学证明。",
        "",
        "新增 `mrta_search/gnn_constructor.py`、`mrta_search/vns.py`、本训练/实验入口和直接测试；生成独立的 `models/gnn_constructor_v1.pt`。旧 GraphCandidateRanker、C2/C4、旧 GAT checkpoint 和候选标签均未用于训练。",
        "",
        "图节点是固定拆分后实际焊接块与四个机器人；边为四近邻、同轨潜在干涉关系和块—合法机器人关系。网络是两层、4 头、hidden=64 的 GAT；静态图每个实例只编码一次，策略头结合机器人累计加工负载、任务数、轨道负载差、同轨邻机负载、最后位置和路线左右边界逐块选择机器人。非法机器人在 softmax/argmax 前置为负无穷。所有方法共用确定性的同轨同向 X 递增路线排序，以及完全相同的有限空间边界 ±1/±2 和路线重排修复。",
        "",
        "VNS 只有 N1 跨机器人迁移、N2 合法交换、N3 单路线 2-opt/重插入。廉价路线代价只用于邻居排序；接受必须依次通过方向 DP、既有 B32 和独立 certifier，且严格降低 Cmax。",
        "",
        "## 训练",
        "",
        f"运行环境：`D:\\pybullet_test\\.venv\\Scripts\\python.exe`，CPU PyTorch；训练 seed={TRAINING_SEED}。专家预算每实例 {args.expert_budget:.1f}s，训练 {args.epochs} epochs，离线训练耗时 {training['training_seconds']:.3f}s。",
        "",
        f"17 个 TRAIN workbook 中得到 {sum(row['status']=='CERTIFIED' for row in train_diag)} 个可认证专家；6 个 DEV workbook 中得到 {sum(row['status']=='CERTIFIED' for row in dev_diag)} 个可认证验证专家。只有 TRAIN 更新权重，DEV 只选 epoch。专家初始/最终 Cmax 均来自本轮固定拆分 RULE+VNS。",
        "",
        "| 专家集合 | 可认证/总数 | 平均初始 Cmax | 平均最终 Cmax | 严格改善实例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, diagnostics in (("TRAIN", train_diag), ("DEV", dev_diag)):
        certified = [row for row in diagnostics if row["status"] == "CERTIFIED"]
        lines.append(
            f"| {label} | {len(certified)}/{len(diagnostics)} | {_fmt(_mean(row['initial_cmax'] for row in certified))} | {_fmt(_mean(row['final_cmax'] for row in certified))} | {sum(row['final_cmax'] < row['initial_cmax'] - 1e-9 for row in certified)} |"
        )
    failed_experts = [
        f"{row['entry']['instance_id']} ({'/'.join(row.get('initial_failures', ()))})"
        for row in (*train_diag, *dev_diag)
        if row["status"] != "CERTIFIED"
    ]
    lines += [
        "",
        "未进入监督数据的专家失败：" + ("；".join(failed_experts) if failed_experts else "无") + "。",
        "",
        "| 指标 | TRAIN | DEV |",
        "| --- | ---: | ---: |",
        f"| 监督步数 | {training['train_metrics']['steps']} | {training['dev_metrics']['steps']} |",
        f"| 交叉熵损失 | {training['train_metrics']['loss']:.6f} | {training['dev_metrics']['loss']:.6f} |",
        f"| 分配准确率 | {100*training['train_metrics']['accuracy']:.2f}% | {100*training['dev_metrics']['accuracy']:.2f}% |",
        f"| 合法动作比例 | {100*training['train_metrics']['legal_action_fraction']:.2f}% | {100*training['dev_metrics']['legal_action_fraction']:.2f}% |",
        f"| 完整覆盖合法率 | {100*training['train_metrics']['coverage_fraction']:.2f}% | {100*training['dev_metrics']['coverage_fraction']:.2f}% |",
        "",
        f"最佳 epoch={training['best_epoch']}。首 epoch train/dev loss={training['history'][0]['train']['loss']:.6f}/{training['history'][0]['dev']['loss']:.6f}；末 epoch={training['history'][-1]['epoch']}，loss={training['history'][-1]['train']['loss']:.6f}/{training['history'][-1]['dev']['loss']:.6f}。",
        "",
        "## Pilot 设置与逐运行结果",
        "",
        f"DEV 只使用下列静态选择的三个实例，每个两个求解 seed；四种主方法预算均为 {args.pilot_budget:.1f}s。17/6 白名单的 DEV 代表实例中没有精确 N85，故 LARGE 使用静态最接近的 N86（target_N=85）。表中“初始 Cmax”是共同有限可行性修复后的首个认证解；修复前失败仍计入失败列。输入读取不计时；GNN 模型加载、构造、修复、VNS、调度和认证均计时。Adapted-WAG+VNS 同预算运行，但可搜索可选 Y/X 拆分，只作外部背景。",
        "",
        *[f"- {_entry_label(entry)}" for entry in pilot_entries],
        "",
        "| 实例 | seed | 方法 | 状态 | 初始 Cmax | 最终 Cmax | 空走 | 等待 | R0/R1/R2/R3 任务数 | R0/R1/R2/R3 负载 | 加载/推理 ms | VNS迭代 | N1/N2/N3严格Cmax改善 | DEADLOCK/方向失败/INFEASIBLE或认证数值失败 |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in pilot:
        final = row["final"]
        failure = row["failures"]
        init_failures = Counter(failure["INITIAL"])
        deadlock = failure.get("DEADLOCK", 0) + init_failures["DEADLOCK"]
        direction = failure.get("DIRECTION_INFEASIBLE", 0) + init_failures["DIRECTION_INFEASIBLE"]
        numeric = failure.get("NUMERIC_FAILURE", 0) + failure.get("INFEASIBLE", 0) + init_failures["NUMERIC_FAILURE"] + init_failures["INFEASIBLE"]
        lines.append(
            "| " + " | ".join(
                [
                    _entry_label(row["entry"]),
                    str(row["seed"]),
                    row["method"],
                    row["status"],
                    _fmt(row["initial"]["cmax"]),
                    _fmt(final["cmax"]),
                    _fmt(final["empty"]),
                    _fmt(final["waiting"]),
                    "—" if final["counts"] is None else "/".join(map(str, final["counts"])),
                    "—" if final["loads"] is None else "/".join(_fmt(value, 1) for value in final["loads"]),
                    f"{1000*row['model_load_seconds']:.2f}/{1000*row['gnn_inference_seconds']:.2f}" if row["method"].startswith("GNN") else "—",
                    str(row["iterations"]),
                    "/".join(str(row["improvements"][name]) for name in NEIGHBORHOODS),
                    f"{deadlock}/{direction}/{numeric}",
                ]
            ) + " |"
        )
    lines += [
        "",
        "## 汇总与问题回答",
        "",
        "| 方法 | 认证成功 | 平均初始 Cmax | 平均最终 Cmax | 相对 RULE+VNS 最终差异 | 平均求解秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rule_vns_mean = _mean(row["final"]["cmax"] for row in pilot if row["method"] == "RULE+VNS")
    summaries = {}
    for method in methods:
        rows = [row for row in pilot if row["method"] == method]
        summary = {
            "success": sum(row["status"] == "CERTIFIED" for row in rows),
            "runs": len(rows),
            "initial": _mean(row["initial"]["cmax"] for row in rows),
            "final": _mean(row["final"]["cmax"] for row in rows),
            "seconds": _mean(row["solve_seconds"] for row in rows),
        }
        summaries[method] = summary
        delta = None if summary["final"] is None or rule_vns_mean is None else 100.0 * (summary["final"] - rule_vns_mean) / rule_vns_mean
        lines.append(
            f"| {method} | {summary['success']}/{summary['runs']} | {_fmt(summary['initial'])} | {_fmt(summary['final'])} | {'—' if delta is None else f'{delta:+.2f}%'} | {_fmt(summary['seconds'])} |"
        )
    contributions = {
        name: sum(row["improvements"][name] for row in pilot if row["method"].endswith("+VNS"))
        for name in NEIGHBORHOODS
    }
    gnn_initial = summaries["GNN"]["initial"]
    rule_initial = summaries["RULE"]["initial"]
    gnn_final = summaries["GNN+VNS"]["final"]
    rule_final = summaries["RULE+VNS"]["final"]
    initial_delta = None if gnn_initial is None or rule_initial is None else 100.0 * (gnn_initial - rule_initial) / rule_initial
    final_delta = None if gnn_final is None or rule_final is None else 100.0 * (gnn_final - rule_final) / rule_final
    gnn_vns_rows = [row for row in pilot if row["method"] == "GNN+VNS" and row["initial"]["cmax"] is not None]
    gnn_vns_improved = sum(row["final"]["cmax"] < row["initial"]["cmax"] - 1e-9 for row in gnn_vns_rows)
    lines += [
        "",
        "### 共享区域与路线空间次序",
        "",
        "| 方法 | 共享块分配总数 R0/R1/R2/R3 | 各机器人相邻任务 X 递增比例均值 |",
        "| --- | --- | --- |",
    ]
    for method in methods:
        rows = [row for row in pilot if row["method"] == method]
        shared = [
            sum(row["assignment"]["shared_counts"][robot] for row in rows)
            for robot in range(4)
        ]
        order = [
            _mean(row["assignment"]["increasing_x_pair_fraction"][robot] for row in rows)
            for robot in range(4)
        ]
        lines.append(
            f"| {method} | {'/'.join(map(str, shared))} | {'/'.join(_fmt(value, 2) for value in order)} |"
        )
    lines += [
        "",
        f"**A：GNN 初始构造。** 平均初始 Cmax 相对 RULE 为 {'无法计算' if initial_delta is None else f'{initial_delta:+.2f}%'}；GNN/RULE 初始认证成功分别为 {summaries['GNN']['success']}/{summaries['GNN']['runs']} 与 {summaries['RULE']['success']}/{summaries['RULE']['runs']}。因此 GNN 构造{'有正收益' if initial_delta is not None and initial_delta < 0 and summaries['GNN']['success'] >= summaries['RULE']['success'] else '没有显示稳定优势'}。",
        "",
        f"**B：相同 VNS 后。** GNN+VNS 相对 RULE+VNS 平均最终 Cmax 为 {'无法计算' if final_delta is None else f'{final_delta:+.2f}%'}。据此，GNN+VNS {'优于' if final_delta is not None and final_delta < 0 and summaries['GNN+VNS']['success'] >= summaries['RULE+VNS']['success'] else '未优于'}相同条件下的 RULE+VNS。",
        "",
        f"**C：网络学到什么。** DEV 教师回放中 GNN/专家一致率 {100*agreement['gnn_expert_accuracy']:.2f}%，规则/专家一致率 {100*agreement['rule_expert_accuracy']:.2f}%；共享覆盖块共 {agreement['shared_steps']} 个，GNN/专家一致率 {100*agreement['shared_gnn_expert_accuracy']:.2f}%。这些指标与逐运行的四机器人负载、共享任务去向和同向 X 空间次序一起用于判断。若 GNN 并未明显超过规则一致率，则现有证据只支持其模仿负载/区域规则，不能声称学到了额外空间协调规律。",
        "",
        f"**D：VNS 贡献。** GNN+VNS 在 {gnn_vns_improved}/{len(gnn_vns_rows)} 个可比较运行中严格降低 Cmax。主方法全部 VNS 运行的有效改善数为 N1={contributions['N1_RELOCATE']}、N2={contributions['N2_SWAP']}、N3={contributions['N3_ROUTE']}；零贡献的邻域仍如实保留。",
        "",
        "### 外部 Adapted-WAG+VNS（不同搜索空间）",
        "",
        "| 实例 | seed | 状态 | Cmax | 求解秒 |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in wag_rows:
        lines.append(
            f"| {_entry_label(row['entry'])} | {row['seed']} | {row['status']} | {_fmt(row['cmax'])} | {_fmt(row['solve_seconds'])} |"
        )
    lines += [
        "",
        "Adapted-WAG+VNS 可改变可选 Y/X 拆分，不能与上述固定拆分四组混称为同一搜索空间下的公平比较。",
        "",
        "## 未解决问题与阶段判断",
        "",
        "B32 的 DEADLOCK 仍只表示固定预算调度未找到方案，不是数学不可行证明。首轮数据量只有 17/6 workbook，且专家本身来自短预算 RULE+VNS，存在明显的小样本和教师上限风险。当前网络只决定机器人归属；拆分点、方向和调度语义均未学习或改变。",
        "",
    ]
    effective = initial_delta is not None and initial_delta < 0 and summaries["GNN"]["success"] >= summaries["RULE"]["success"]
    stable_vns = bool(gnn_vns_rows) and gnn_vns_improved >= math.ceil(len(gnn_vns_rows) / 2)
    superior = final_delta is not None and final_delta < 0 and summaries["GNN+VNS"]["success"] >= summaries["RULE+VNS"]["success"]
    lines += [
        f"- GNN 构造是否有效：**{'是（本 pilot 有正向证据）' if effective else '否（本 pilot 未显示可靠优势）'}**。",
        f"- VNS 是否稳定改善 GNN 初值：**{'是' if stable_vns else '否'}**。",
        f"- GNN+VNS 是否优于 RULE+VNS：**{'是' if superior else '否'}**。",
        f"- 是否已有证据进入可选 Y/X 拆分神经决策：**{'可考虑，但仍需更大独立验证' if effective and superior else '没有；应先解决当前任务分配/路线协调限制'}**。",
        "",
        "上述判断只针对本轮固定拆分 DEV pilot，不外推到封存 ID_TEST 或完整 V2 可选拆分问题。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-budget", type=float, default=2.0)
    parser.add_argument("--pilot-budget", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-wag", action="store_true")
    parser.add_argument(
        "--vns-fix-mode",
        action="store_true",
        help="skip training and compare legacy/new VNS selection from paired initials",
    )
    parser.add_argument("--vns-fix-budget", type=float, default=20.0)
    parser.add_argument("--vns-fix-all-dev", action="store_true")
    parser.add_argument("--vns-fix-seeds", type=int, nargs="+", default=[20261022])
    args = parser.parse_args()
    config = ScientificConfig()
    representatives = ranker_dataset.representatives_v05()
    train_entries = [entry for entry in representatives if entry["split"] == "MLP_TRAIN"]
    dev_entries = [entry for entry in representatives if entry["split"] == "MLP_DEV"]
    if (len(train_entries), len(dev_entries)) != (17, 6):
        raise ValueError("expected existing 17 TRAIN / 6 DEV workbook split")
    if args.vns_fix_mode:
        run_vns_fix_mode(
            dev_entries,
            config,
            args.vns_fix_budget,
            args.vns_fix_all_dev,
            tuple(args.vns_fix_seeds),
        )
        return
    train_records, train_diag = build_experts(train_entries, config, args.expert_budget)
    dev_records, dev_diag = build_experts(dev_entries, config, args.expert_budget)
    if not train_records or not dev_records:
        raise RuntimeError("no certified expert data available")
    model, training = train_model(train_records, dev_records, config, args.epochs)
    agreement = _rule_agreement(dev_records, model, config)
    pilot_entries = _select_pilot_entries(dev_entries)
    pilot = []
    for entry in pilot_entries:
        for seed in PILOT_SEEDS:
            for method in ("RULE", "RULE+VNS", "GNN", "GNN+VNS"):
                print(f"pilot {_entry_label(entry)} seed={seed} {method}", flush=True)
                pilot.append(run_method(entry, method, seed, config, args.pilot_budget))
    wag_rows = []
    if not args.skip_wag:
        for entry in pilot_entries:
            for seed in PILOT_SEEDS:
                print(f"external WAG {_entry_label(entry)} seed={seed}", flush=True)
                wag_rows.append(run_wag_reference(entry, seed, config, args.pilot_budget))
    write_report(
        training,
        train_diag,
        dev_diag,
        pilot,
        wag_rows,
        pilot_entries,
        agreement,
        args,
    )
    print(f"checkpoint: {MODEL_PATH}", flush=True)
    print(f"report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

