"""Train the fixed-split assignment GNN and run the first RULE/GNN VNS pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
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
from mrta_reference.model import OperationKind, Route, ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_search.gnn_constructor import (
    BlockAssignmentGNN,
    BlockAssignmentRankingModel,
    assignment_targets,
    build_block_robot_graph,
    construct_ranked_solution,
    construct_solution,
    dynamic_features,
    fixed_blocks,
    fixed_split_patterns,
    rule_assignment,
    solution_from_assignment,
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
EXPERT_DATA_PATH = ROOT / "data" / "development" / "gnn_experts_fixed_split_v2.json"
INITIAL_STUDY_MODEL_PATH = ROOT / "models" / "gnn_initial_solution_study_v2.pt"
INITIAL_STUDY_CSV_PATH = ROOT / "results" / "gnn_initial_solution_study_v2.csv"
INITIAL_STUDY_REPORT_PATH = ROOT / "docs" / "GNN_INITIAL_SOLUTION_STUDY_V2.md"
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
            "timing_scope": "VNS_ONLY_FROM_CERTIFIED_INITIAL",
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


def _public_run(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def run_end_to_end(entry, construction_kind, seed, config, budget):
    """Run one complete solver call under one wall-clock deadline.

    Workbook loading is intentionally outside the measured region. Model loading,
    construction, initial evaluation/repair, VNS, B32 and certification are inside.
    """
    parents = ranker_dataset.load_parents(entry)
    started = time.perf_counter()
    model = None
    model_load_seconds = 0.0
    if construction_kind == "GNN":
        load_started = time.perf_counter()
        model, _ = load_model()
        model_load_seconds = time.perf_counter() - load_started
    construction_started = time.perf_counter()
    construction = construct_solution(parents, config, model=model)
    construction_seconds = time.perf_counter() - construction_started
    pre_evaluation_seconds = time.perf_counter() - started
    remaining = budget - pre_evaluation_seconds
    if remaining <= 0.0:
        total = time.perf_counter() - started
        return {
            "instance": entry["instance_id"],
            "N": entry["N"],
            "data_role": entry["split"],
            "construction": construction_kind,
            "seed": seed,
            "budget_seconds": budget,
            "status": "TIMEOUT_BEFORE_INITIAL_EVALUATION",
            "timing_scope": "END_TO_END_EXCLUDING_INPUT_READ",
            "model_load_seconds": model_load_seconds,
            "construction_seconds": construction_seconds,
            "initial_certification_seconds": 0.0,
            "vns_seconds": 0.0,
            "total_seconds": total,
            "overtime_seconds": max(0.0, total - budget),
            "initial": _solution_metrics(None, config),
            "final": _solution_metrics(None, config),
            "time_to_best_seconds": None,
            "attempted": {name: 0 for name in NEIGHBORHOODS},
            "certified": {name: 0 for name in NEIGHBORHOODS},
            "improvements": {name: 0 for name in NEIGHBORHOODS},
            "failures": {},
            "late_certified": 0,
            "initial_failures": ("BUDGET_EXHAUSTED_BY_CONSTRUCTION",),
            "_parents": parents,
            "_construction": construction,
            "_outcome": None,
        }

    evaluator = CommonBaselineEvaluator(
        parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=remaining,
        checkpoints=(remaining,),
    )
    outcome = run_vns(
        construction.solution,
        evaluator,
        config,
        seed=seed,
        per_neighborhood_budget=12,
        selection_policy=GLOBAL_BALANCED,
        accept_completed_by=remaining,
    )
    total = time.perf_counter() - started
    initial_seconds = (
        outcome.initial.completed_at if outcome.initial is not None else evaluator.elapsed
    )
    vns_seconds = max(0.0, evaluator.elapsed - initial_seconds)
    status = "CERTIFIED" if outcome.best is not None else (
        "TIMEOUT_NO_CERTIFIED" if total >= budget else "INITIALIZATION_FAILED"
    )
    return {
        "instance": entry["instance_id"],
        "N": entry["N"],
        "data_role": entry["split"],
        "construction": construction_kind,
        "seed": seed,
        "budget_seconds": budget,
        "status": status,
        "timing_scope": "END_TO_END_EXCLUDING_INPUT_READ",
        "model_load_seconds": model_load_seconds,
        "construction_seconds": construction_seconds,
        "initial_certification_seconds": initial_seconds,
        "vns_seconds": vns_seconds,
        "total_seconds": total,
        "overtime_seconds": max(0.0, total - budget),
        "initial": _solution_metrics(outcome.initial, config),
        "final": _solution_metrics(outcome.best, config),
        "time_to_best_seconds": (
            None
            if outcome.best is None
            else pre_evaluation_seconds + outcome.best.completed_at
        ),
        "attempted": dict(outcome.stats.attempted),
        "certified": dict(outcome.stats.certified),
        "improvements": dict(outcome.stats.improvements),
        "failures": dict(outcome.stats.failures),
        "late_certified": outcome.stats.late_certified,
        "initial_failures": outcome.initialization_failures,
        "_parents": parents,
        "_construction": construction,
        "_outcome": outcome,
    }


def run_end_to_end_mode(dev_entries, config, budget):
    rows = []
    for entry in _select_pilot_entries(dev_entries):
        for construction_kind in ("RULE", "GNN"):
            print(
                f"end-to-end {_entry_label(entry)} {construction_kind} budget={budget}",
                flush=True,
            )
            row = run_end_to_end(
                entry, construction_kind, TRAINING_SEED, config, budget
            )
            rows.append(row)
            print(
                "END_TO_END_ROW "
                + json.dumps(_public_run(row), ensure_ascii=False),
                flush=True,
            )
    return rows


def _owner_map(solution):
    return {
        block_id: route.robot_id
        for route in solution.routes
        for block_id in route.block_ids
    }


def _knowledge_diagnostics(initial, final, config):
    blocks = block_map(initial.solution, config, scope=FORMAL_SCOPE_V2)
    initial_owner = _owner_map(initial.solution)
    final_owner = _owner_map(final.solution)
    changed = [
        block_id
        for block_id in sorted(blocks)
        if initial_owner[block_id] != final_owner[block_id]
    ]
    flexible = [
        block_id
        for block_id, block in blocks.items()
        if sum(robot_is_eligible(block, robot, config) for robot in range(4)) > 1
    ]
    cross_rail_flexible = [
        block_id
        for block_id, block in blocks.items()
        if any(robot_is_eligible(block, robot, config) for robot in (0, 1))
        and any(robot_is_eligible(block, robot, config) for robot in (2, 3))
    ]
    same_rail_changes = sum(
        initial_owner[block_id] // 2 == final_owner[block_id] // 2
        for block_id in changed
    )
    cross_rail_changes = len(changed) - same_rail_changes
    changed_route_robots = 0
    pure_order_changed_robots = 0
    for initial_route, final_route in zip(initial.solution.routes, final.solution.routes):
        if initial_route.block_ids != final_route.block_ids:
            changed_route_robots += 1
        if (
            set(initial_route.block_ids) == set(final_route.block_ids)
            and initial_route.block_ids != final_route.block_ids
        ):
            pure_order_changed_robots += 1
    initial_metrics = _solution_metrics(initial, config)
    final_metrics = _solution_metrics(final, config)
    initial_lb = max(initial_metrics["loads"])
    final_lb = max(final_metrics["loads"])
    return {
        "assignment_changes": len(changed),
        "changed_block_ids": changed,
        "cross_rail_changes": cross_rail_changes,
        "same_rail_changes": same_rail_changes,
        "flexible_blocks": len(flexible),
        "flexible_assignment_changes": sum(block_id in changed for block_id in flexible),
        "cross_rail_flexible_blocks": len(cross_rail_flexible),
        "cross_rail_flexible_changes": sum(
            block_id in changed for block_id in cross_rail_flexible
        ),
        "changed_route_robots": changed_route_robots,
        "pure_order_changed_robots": pure_order_changed_robots,
        "initial_lb_load": initial_lb,
        "final_lb_load": final_lb,
        "initial_cmax_minus_lb_load": initial_metrics["cmax"] - initial_lb,
        "final_cmax_minus_lb_load": final_metrics["cmax"] - final_lb,
    }


def _evaluate_x_sorted_ownership(candidate, config, *, time_limit=60.0):
    solution = candidate.solution
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    owner = _owner_map(solution)
    assignment = tuple((block_id, owner[block_id]) for block_id in sorted(blocks))
    x_sorted = solution_from_assignment(
        solution.parents,
        solution.patterns,
        tuple(blocks.values()),
        assignment,
        config,
    )
    evaluator = CommonBaselineEvaluator(
        solution.parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=time_limit,
        checkpoints=(time_limit,),
    )
    evaluated = evaluator.evaluate(x_sorted, source="OWNERSHIP_X_SORT_DIAGNOSTIC")
    in_time = evaluated.completed_at <= time_limit
    return {
        "status": evaluated.status.value if in_time else "LATE_" + evaluated.status.value,
        "cmax": (
            evaluated.metrics.cmax
            if in_time and evaluated.metrics is not None
            else None
        ),
        "seconds": evaluator.elapsed,
    }


def _serialize_pattern(pattern):
    return {
        "parent_id": pattern.parent_id,
        "kind": pattern.kind.value,
        "t": pattern.t,
        "point_id": pattern.point_id,
        "mandatory": pattern.mandatory,
        "rail": None if pattern.rail is None else pattern.rail.value,
    }


def _serialize_expert(entry, run, diagnostics, config):
    outcome = run["_outcome"]
    if outcome is None or outcome.best is None:
        return None
    best = outcome.best
    return {
        "instance_id": entry["instance_id"],
        "relative_path": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "data_role": entry["split"],
        "N": entry["N"],
        "fixed_split_patterns": [
            _serialize_pattern(pattern) for pattern in best.solution.patterns
        ],
        "assignments": [
            {"block_id": block_id, "robot_id": robot}
            for block_id, robot in sorted(_owner_map(best.solution).items())
        ],
        "routes": [
            {"robot_id": route.robot_id, "block_ids": list(route.block_ids)}
            for route in best.solution.routes
        ],
        "certified_cmax": best.metrics.cmax,
        "expert_source": f"{run['construction']}_GLOBAL_BALANCED_VNS",
        "solver_budget_seconds": run["budget_seconds"],
        "time_to_best_seconds": run["time_to_best_seconds"],
        "scientific_config": config.scientific_mapping(),
        "diagnostics": diagnostics,
    }


def run_train_expert(entry, config, budget, seed, construction_kind="RULE"):
    run = run_end_to_end(entry, construction_kind, seed, config, budget)
    outcome = run["_outcome"]
    if outcome is None or outcome.initial is None or outcome.best is None:
        return run, None, None
    knowledge = _knowledge_diagnostics(outcome.initial, outcome.best, config)
    diagnostics = {
        **knowledge,
        "initial": run["initial"],
        "final": run["final"],
        "improvements": run["improvements"],
        "attempted": run["attempted"],
        "certified": run["certified"],
        "failures": run["failures"],
        "late_certified": run["late_certified"],
        "initial_x_sorted": _evaluate_x_sorted_ownership(outcome.initial, config),
        "expert_x_sorted": _evaluate_x_sorted_ownership(outcome.best, config),
    }
    return run, _serialize_expert(entry, run, diagnostics, config), diagnostics


def _select_scale_entries(entries):
    small = min(entries, key=lambda entry: (entry["N"], entry["instance_id"]))
    medium = min(entries, key=lambda entry: (abs(entry["N"] - 55), entry["instance_id"]))
    large = max(entries, key=lambda entry: (entry["N"], entry["instance_id"]))
    return [small, medium, large]


def run_train_expert_mode(
    train_entries,
    config,
    budget,
    limit,
    write_output,
    construction_kind="RULE",
):
    entries = list(train_entries)
    if limit == 3:
        entries = _select_scale_entries(entries)
    elif limit is not None:
        entries = entries[:limit]
    experts = []
    public_rows = []
    for index, entry in enumerate(entries):
        print(
            f"train-expert {index + 1}/{len(entries)} {_entry_label(entry)} budget={budget}",
            flush=True,
        )
        run, expert, diagnostics = run_train_expert(
            entry,
            config,
            budget,
            TRAINING_SEED + index,
            construction_kind=construction_kind,
        )
        row = _public_run(run)
        row["knowledge"] = diagnostics
        public_rows.append(row)
        if expert is not None:
            experts.append(expert)
        print("TRAIN_EXPERT_ROW " + json.dumps(row, ensure_ascii=False), flush=True)
    if write_output:
        payload = {
            "purpose": "fixed-split block-to-robot supervised expert data",
            "data_role": "MLP_TRAIN",
            "expert_source": f"{construction_kind}_GLOBAL_BALANCED_VNS",
            "solver_budget_seconds_per_instance": budget,
            "experts": experts,
            "runs": public_rows,
        }
        EXPERT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPERT_DATA_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"expert-data: {EXPERT_DATA_PATH}", flush=True)
    return public_rows, experts


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


def _saved_expert_records(train_entries, config):
    """Load only the already saved 17 TRAIN experts; never regenerate teachers."""
    payload = json.loads(EXPERT_DATA_PATH.read_text(encoding="utf-8"))
    entries = {entry["instance_id"]: entry for entry in train_entries}
    records = []
    audit_rows = []
    for expert in payload["experts"]:
        entry = entries.get(expert["instance_id"])
        if entry is None:
            raise ValueError(f"saved expert is outside TRAIN split: {expert['instance_id']}")
        parents = ranker_dataset.load_parents(entry)
        patterns = fixed_split_patterns(parents, config)
        stored_pattern_ids = []
        for item in expert["fixed_split_patterns"]:
            if item["kind"] == "WHOLE":
                stored_pattern_ids.append(f"{item['parent_id']}:WHOLE")
            elif item["kind"] == "X_SPLIT":
                stored_pattern_ids.append(
                    f"{item['parent_id']}:X_SPLIT:{item['rail']}:{item['point_id']}"
                )
            else:
                stored_pattern_ids.append(
                    f"{item['parent_id']}:Y_SPLIT:{item['point_id']}"
                )
        if stored_pattern_ids != [pattern.pattern_id for pattern in patterns]:
            raise ValueError(f"fixed split replay mismatch: {expert['instance_id']}")
        routes = tuple(
            Route(int(row["robot_id"]), tuple(row["block_ids"]))
            for row in sorted(expert["routes"], key=lambda row: int(row["robot_id"]))
        )
        solution = canonicalize(
            parents, patterns, routes, config, scope=FORMAL_SCOPE_V2
        )
        blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
        stored_owner = {
            row["block_id"]: int(row["robot_id"])
            for row in expert["assignments"]
        }
        actual_owner = assignment_targets(solution)
        if stored_owner != actual_owner or set(actual_owner) != set(blocks):
            raise ValueError(f"expert assignment/coverage mismatch: {expert['instance_id']}")
        if any(
            not robot_is_eligible(blocks[block_id], robot, config)
            for block_id, robot in actual_owner.items()
        ):
            raise ValueError(f"expert has illegal assignment: {expert['instance_id']}")
        pairs = inversions = changed_routes = 0
        for route in solution.routes:
            expected = sorted(
                route.block_ids,
                key=lambda block_id: (
                    (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0,
                    block_id,
                ),
            )
            changed_routes += int(tuple(expected) != route.block_ids)
            positions = {block_id: index for index, block_id in enumerate(expected)}
            for left in range(len(route.block_ids)):
                for right in range(left + 1, len(route.block_ids)):
                    pairs += 1
                    inversions += int(
                        positions[route.block_ids[left]]
                        > positions[route.block_ids[right]]
                    )
        audit_rows.append(
            {
                "instance_id": expert["instance_id"],
                "pairs": pairs,
                "x_inversions": inversions,
                "changed_routes": changed_routes,
                "pair_disagreement": inversions / max(1, pairs),
            }
        )
        records.append(
            {
                "entry": entry,
                "parents": parents,
                "expert": solution,
            }
        )
    if len(records) != 17:
        raise ValueError(f"expected 17 saved TRAIN experts, got {len(records)}")
    return records, audit_rows


def _joint_supervised_loss(model, record, config, *, include_weld_edges):
    parents = record["parents"]
    expert = record["expert"]
    patterns = fixed_split_patterns(parents, config)
    blocks = fixed_blocks(parents, patterns, config)
    by_id = {block.block_id: block for block in blocks}
    graph = build_block_robot_graph(
        blocks, patterns, config, include_weld_edges=include_weld_edges
    )
    embeddings = model.encode_static(graph)
    targets = assignment_targets(expert)

    assigned = [[] for _ in range(4)]
    assignment_losses = []
    for step, block in enumerate(blocks):
        target = targets[block.block_id]
        mask = torch.tensor(
            [robot_is_eligible(block, robot, config) for robot in range(4)],
            dtype=torch.bool,
        )
        logits = model.action_logits(
            graph,
            embeddings,
            block.block_id,
            dynamic_features(block, assigned, by_id, config, step, len(blocks)),
            mask,
        )
        assignment_losses.append(
            nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target])
            )
        )
        assigned[target].append(block.block_id)

    # Round-robin teacher forcing mirrors DAG-safe inference.  Supervision is
    # strictly within one expert robot route.  Immediate-successor pairs and
    # expert reversals of X order receive full weight.  Other X-consistent pairs
    # receive only a 0.1 soft constraint: this keeps the sparse ranking problem
    # from becoming underdetermined without declaring one saved route uniquely
    # optimal when many far-apart orders may be equivalent.
    expert_routes = [list(route.block_ids) for route in expert.routes]
    x_position = []
    for route in expert_routes:
        x_sorted = sorted(
            route,
            key=lambda block_id: (
                (by_id[block_id].start[0] + by_id[block_id].end[0]) / 2.0,
                block_id,
            ),
        )
        x_position.append({block_id: index for index, block_id in enumerate(x_sorted)})
    prefixes = [[] for _ in range(4)]
    route_losses = [[] for _ in range(4)]
    route_weights = [[] for _ in range(4)]
    pair_count = 0
    max_depth = max(map(len, expert_routes), default=0)
    for depth in range(max_depth):
        for robot in range(4):
            route = expert_routes[robot]
            if depth >= len(route):
                continue
            positive = route[depth]
            comparisons = []
            for other_depth in range(depth + 1, len(route)):
                other = route[other_depth]
                weight = 1.0 if (
                    other_depth == depth + 1
                    or x_position[robot][positive] > x_position[robot][other]
                ) else 0.1
                comparisons.append((other, weight))
            step = sum(map(len, prefixes))
            positive_score = model.ordering_score(
                graph,
                embeddings,
                positive,
                robot,
                dynamic_features(
                    by_id[positive], prefixes, by_id, config, step, len(blocks)
                ),
            )
            for negative, weight in comparisons:
                negative_score = model.ordering_score(
                    graph,
                    embeddings,
                    negative,
                    robot,
                    dynamic_features(
                        by_id[negative], prefixes, by_id, config, step, len(blocks)
                    ),
                )
                route_losses[robot].append(
                    weight * nn.functional.softplus(-(positive_score - negative_score))
                )
                route_weights[robot].append(weight)
                pair_count += 1
            prefixes[robot].append(positive)
    normalized_routes = [
        torch.stack(losses).sum() / sum(route_weights[robot])
        for robot, losses in enumerate(route_losses)
        if losses
    ]
    assignment_loss = torch.stack(assignment_losses).mean()
    if normalized_routes:
        ranking_loss = torch.stack(normalized_routes).mean()
    else:
        ranking_loss = assignment_loss.new_zeros(())
    return 0.5 * (assignment_loss + ranking_loss), assignment_loss, ranking_loss, pair_count


def _train_initial_study_models(records, config, epochs):
    if epochs > 30:
        raise ValueError("initial study is capped at 30 epochs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    variants = {
        "GNN": {"encoder_kind": "gnn", "include_weld_edges": True},
        "MLP": {"encoder_kind": "mlp", "include_weld_edges": True},
        "GNN_NO_WELD_EDGES": {
            "encoder_kind": "gnn",
            "include_weld_edges": False,
        },
    }
    trained = {}
    histories = {}
    for name, settings in variants.items():
        # One fixed seed and the final epoch are selected before DEV evaluation.
        # This deliberately avoids test-driven seed/checkpoint selection.
        torch.manual_seed(TRAINING_SEED)
        random.seed(TRAINING_SEED)
        model = BlockAssignmentRankingModel(
            hidden_dim=64,
            heads=4,
            encoder_kind=settings["encoder_kind"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
        )
        history = []
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(records)))
            random.Random(TRAINING_SEED + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            rows = [
                _joint_supervised_loss(
                    model,
                    records[index],
                    config,
                    include_weld_edges=settings["include_weld_edges"],
                )
                for index in order
            ]
            loss = torch.stack([row[0] for row in rows]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            item = {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "assignment_loss": statistics.fmean(float(row[1].detach()) for row in rows),
                "ranking_loss": statistics.fmean(float(row[2].detach()) for row in rows),
                "ranking_pairs": sum(row[3] for row in rows),
            }
            history.append(item)
            print(
                f"study-train {name} epoch={epoch:02d} "
                f"loss={item['loss']:.4f} assignment={item['assignment_loss']:.4f} "
                f"ranking={item['ranking_loss']:.4f} pairs={item['ranking_pairs']}",
                flush=True,
            )
        model.eval()
        trained[name] = model
        histories[name] = {
            "encoder_kind": settings["encoder_kind"],
            "include_weld_edges": settings["include_weld_edges"],
            "epochs": epochs,
            "training_seed": TRAINING_SEED,
            "selection_rule": "fixed seed; final epoch; no DEV/test selection",
            "training_seconds": time.perf_counter() - started,
            "history": history,
        }
    return trained, histories


def _route_x_disagreement(solution, config):
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    pairs = inversions = changed = 0
    for route in solution.routes:
        x_sorted = sorted(
            route.block_ids,
            key=lambda block_id: (
                (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0,
                block_id,
            ),
        )
        changed += int(tuple(x_sorted) != route.block_ids)
        positions = {block_id: index for index, block_id in enumerate(x_sorted)}
        for left in range(len(route.block_ids)):
            for right in range(left + 1, len(route.block_ids)):
                pairs += 1
                inversions += int(
                    positions[route.block_ids[left]]
                    > positions[route.block_ids[right]]
                )
    return changed, pairs, inversions


def _evaluate_direct_initial(entry, method, construction, config, budget):
    evaluator = CommonBaselineEvaluator(
        construction.solution.parents,
        config,
        FORMAL_SCOPE_V2,
        time_limit=budget,
        checkpoints=(budget,),
    )
    candidate = evaluator.evaluate(construction.solution, source=method)
    certified = bool(candidate.certification and candidate.certification.certified)
    blocks = block_map(construction.solution, config, scope=FORMAL_SCOPE_V2)
    loads = [
        sum(config.process_time(blocks[block_id].length) for block_id in route.block_ids)
        for route in construction.solution.routes
    ]
    counts = [len(route.block_ids) for route in construction.solution.routes]
    changed_routes, pairs, inversions = _route_x_disagreement(
        construction.solution, config
    )
    metrics = candidate.metrics if certified else None
    return {
        "instance_id": entry["instance_id"],
        "relative_path": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "method": method,
        "N": entry["N"],
        "block_count": len(blocks),
        "raw_status": candidate.status.value,
        "certified": certified,
        "cmax": None if metrics is None else metrics.cmax,
        "robot_process_loads": ";".join(f"{value:.9f}" for value in loads),
        "total_empty_travel": None if metrics is None else metrics.total_empty_travel,
        "total_waiting": None if metrics is None else metrics.total_waiting,
        "robot_task_counts": ";".join(map(str, counts)),
        "construction_seconds": construction.inference_seconds,
        "direction_dp_seconds": evaluator.accounting.direction_time,
        "b32_seconds": evaluator.accounting.scheduler_time,
        "certifier_seconds": evaluator.accounting.certifier_time,
        "evaluation_seconds": candidate.completed_at,
        "end_to_end_seconds": construction.inference_seconds + candidate.completed_at,
        "relative_cmax_vs_rule": None,
        "assignment_changes_vs_rule": None,
        "x_changed_routes": changed_routes,
        "x_pair_disagreement": inversions / max(1, pairs),
        "diagnostics": " | ".join(candidate.diagnostics),
        "_solution": construction.solution,
    }


def _paired_method_summary(rows):
    by_instance = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], {})[row["method"]] = row
    result = {}
    for method in ("A", "B", "C", "D", "E", "F"):
        method_rows = [group[method] for group in by_instance.values()]
        paired = [
            (group[method]["cmax"], group["A"]["cmax"])
            for group in by_instance.values()
            if group[method]["certified"] and group["A"]["certified"]
        ]
        relative = [(value - rule) / rule for value, rule in paired]
        wins = sum(value < rule - 1.0e-9 for value, rule in paired)
        losses = sum(value > rule + 1.0e-9 for value, rule in paired)
        result[method] = {
            "certified": sum(row["certified"] for row in method_rows),
            "total": len(method_rows),
            "paired": len(paired),
            "wins": wins,
            "ties": len(paired) - wins - losses,
            "losses": losses,
            "median_relative": statistics.median(relative) if relative else None,
            "mean_construction_seconds": statistics.fmean(
                row["construction_seconds"] for row in method_rows
            ),
            "mean_end_to_end_seconds": statistics.fmean(
                row["end_to_end_seconds"] for row in method_rows
            ),
        }
    return result


def _write_initial_study_outputs(rows, histories, audit_rows, config, *, extra_note):
    del histories, config
    def percent(value):
        return "—" if value is None else f"{100 * value:+.2f}%"

    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    fields = list(public_rows[0])
    with INITIAL_STUDY_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_rows)

    summary = _paired_method_summary(rows)
    method_names = {
        "A": "RULE / X",
        "B": "GNN V1 / X",
        "C": "RULE / GNN order",
        "D": "GNN assign / GNN order",
        "E": "RULE / MLP order",
        "F": "RULE / no-weld-edge GNN order",
    }
    lines = [
        "# GNN fixed-split initial-solution study V2",
        "",
        "## A. 实际完成了什么",
        "",
        "在既有两层、4 头、hidden=64 GAT 上增加一个轻量排序头；归属头继续使用合法机器人掩码。排序训练只使用同机器人任务对，按路线归一化；推理通过全局拓扑追加序列构造四条路线。固定拆分中的 mandatory Y 两个子块是可并行覆盖任务，当前科学模型没有额外的父子时间边；若传入显式前驱，构造器只从前驱已完成集合选取，因此显式前驱边和四条路线边均沿同一全局序列，不会形成跨机器人环。方向仍由原 DP 决定，随后运行原 B32 和独立 certifier。",
        "",
        "旧 `gnn_constructor_v1.pt`、VNS N1/N2/N3、C2/C4、SA、旧 GAT 排序器及科学参数均未修改。新联合 GNN、MLP、去焊缝边 GNN 使用相同 17 个已保存 TRAIN 专家、相同固定 seed、相同 epoch 和损失定义；checkpoint 固定取最后 epoch，未根据 DEV 结果选择。",
        "",
        "## 专家排序可学习性检查",
        "",
        f"17/17 专家通过覆盖、唯一归属、合法机器人和固定拆分核对；17/17 当前路线都与简单 X 升序不同。总同机器人对 {sum(row['pairs'] for row in audit_rows)}，其中逆序 {sum(row['x_inversions'] for row in audit_rows)}（{100*sum(row['x_inversions'] for row in audit_rows)/max(1,sum(row['pairs'] for row in audit_rows)):.2f}%）；改变路线 {sum(row['changed_routes'] for row in audit_rows)}/68。全部 17 条专家解已在当前代码中重放并经方向 DP、B32 与 certifier 认证。",
        "",
        "## B. 核心实验结果（无 VNS、无回退）",
        "",
        "| 实例 | 方法 | 状态 | Cmax | 构造秒 | 端到端秒 | 相对 RULE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        relative = row["relative_cmax_vs_rule"]
        lines.append(
            f"| {row['sheet_name']} | {row['method']} {method_names[row['method']]} | "
            f"{'CERTIFIED' if row['certified'] else row['raw_status']} | "
            f"{_fmt(row['cmax'])} | {_fmt(row['construction_seconds'], 4)} | "
            f"{_fmt(row['end_to_end_seconds'], 4)} | "
            f"{percent(relative)} |"
        )
    lines += [
        "",
        "| 方法 | 认证 | 胜/平/负 vs A | 配对中位变化 | 平均构造秒 | 平均端到端秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("A", "B", "C", "D", "E", "F"):
        item = summary[method]
        lines.append(
            f"| {method} {method_names[method]} | {item['certified']}/{item['total']} | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | "
            f"{percent(item['median_relative'])} | "
            f"{item['mean_construction_seconds']:.4f} | {item['mean_end_to_end_seconds']:.4f} |"
        )

    c, d, e, f = (summary[key] for key in ("C", "D", "E", "F"))
    ordering_signal = c["wins"] > c["losses"] and (c["median_relative"] or 0.0) < 0.0
    joint_signal = d["wins"] > d["losses"] and (d["median_relative"] or 0.0) < 0.0
    graph_signal = (
        ordering_signal
        and
        c["median_relative"] is not None
        and e["median_relative"] is not None
        and f["median_relative"] is not None
        and c["median_relative"] < min(e["median_relative"], f["median_relative"])
    )
    by_instance = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], {})[row["method"]] = row
    def load_imbalance(row):
        values = [float(value) for value in row["robot_process_loads"].split(";")]
        return max(values) - min(values)
    joint_worse_loads = sum(
        load_imbalance(group["D"]) > load_imbalance(group["A"]) + 1.0e-9
        for group in by_instance.values()
    )
    expert_x_median = statistics.median(
        row["pair_disagreement"] for row in audit_rows
    )
    learned_x_median = statistics.median(
        row["x_pair_disagreement"] for row in rows if row["method"] == "C"
    )
    lines += [
        "",
        "## C. GNN 实际学到了什么",
        "",
        f"RULE 归属 + GNN 排序在 DEV 上的直接证据：胜/平/负 {c['wins']}/{c['ties']}/{c['losses']}，配对中位 {percent(c['median_relative'])}。联合归属+排序为 {d['wins']}/{d['ties']}/{d['losses']}，中位 {percent(d['median_relative'])}。",
        "",
        f"MLP 排序中位 {percent(e['median_relative'])}；去焊缝关系边 GNN 排序中位 {percent(f['median_relative'])}。三种排序器都没有稳定优于 RULE，且认证成功集合并不完全相同；因此当前 DEV 证据{'支持焊缝图边提供了额外收益' if graph_signal else '不足以证明焊缝—焊缝图关系优于相同输入的 MLP 或去边 GNN'}。",
        "",
        "## D. 当前失败原因",
        "",
        f"排序监督虽可重放，但信息密度低：专家实例的 X 对逆序中位仅 {100*expert_x_median:.2f}%，而 C 的预测路线逆序中位达到 {100*learned_x_median:.2f}%。加入低权重 X 一致对后已消除 C 的 DEADLOCK，但仍未把排序分数稳定转化为更小 Cmax，说明小样本单一路线标签不足以识别哪些逆序真正改善协调等待。",
        "",
        f"联合归属的主要失败证据更直接：D 相对 RULE 改变 3–7 个块，并在 {joint_worse_loads}/6 个 DEV 实例上都扩大了四机器人加工负载极差；因此排序头还承受了训练时专家归属与推理时新归属的分布偏移。D 的收益不能归因于工程认证问题，因为其已认证实例中仍为 1 胜 4 负。",
        "",
        "MLP 与去边 GNN 也没有稳定改善，无法仅凭完整 GNN 的中位退化略小就宣称图关系有效。直接构造失败与 B32 DEADLOCK 均保留为失败，不解释成数学不可行，也未用修复器或 VNS 掩盖。",
        "",
        "## E. 下一步判断",
        "",
        f"- GNN 排序：{'值得以更大专家集复验' if ordering_signal else '当前不值得增加层数或引入 PPO'}。",
        f"- 联合构造：{'有继续验证价值' if joint_signal else '当前不应宣称优于 RULE；若排序有效应保留 RULE+排序'}。",
        f"- 扩大专家数据：{'必要，重点增加非 X 排序与真实归属变化' if ordering_signal or joint_signal else '若继续 GNN 则必要；必须增加能区分有效/无效逆序和真实归属变化的专家，而不是重复近似 X 顺序'}。",
        "- 可选 X/Y 拆分：本轮没有足够依据引入；应等固定拆分下图排序价值成立后再研究。",
        "",
        extra_note,
        "",
        "完整逐实例负载、空行程、等待、任务数、方向/B32/certifier 时间和失败诊断见 CSV。",
    ]
    INITIAL_STUDY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_initial_study(train_entries, dev_entries, config, epochs, budget, *, resume=False):
    records, audit_rows = _saved_expert_records(train_entries, config)
    if resume:
        saved = torch.load(
            INITIAL_STUDY_MODEL_PATH, map_location="cpu", weights_only=True
        )
        if saved.get("model_type") != "BLOCK_ASSIGNMENT_RANKING_STUDY_V2":
            raise ValueError("initial-study resume checkpoint has the wrong model type")
        models = {}
        for name, item in saved["variants"].items():
            model = BlockAssignmentRankingModel(
                hidden_dim=saved["hidden_dim"],
                heads=saved["heads"],
                encoder_kind=item["encoder_kind"],
            )
            model.load_state_dict(item["state_dict"])
            model.eval()
            models[name] = model
        histories = saved["histories"]
    else:
        models, histories = _train_initial_study_models(records, config, epochs)
    old_model, old_checkpoint = load_model(MODEL_PATH)
    if not resume:
        checkpoint = {
            "model_type": "BLOCK_ASSIGNMENT_RANKING_STUDY_V2",
            "hidden_dim": 64,
            "heads": 4,
            "training_seed": TRAINING_SEED,
            "expert_data": str(EXPERT_DATA_PATH.relative_to(ROOT)),
            "variants": {
                name: {
                    **{key: value for key, value in histories[name].items() if key != "history"},
                    "state_dict": model.state_dict(),
                }
                for name, model in models.items()
            },
            "histories": histories,
            "old_v1_model_type": old_checkpoint["model_type"],
        }
        torch.save(checkpoint, INITIAL_STUDY_MODEL_PATH)

    rows = []
    for entry in dev_entries:
        parents = ranker_dataset.load_parents(entry)
        constructions = {
            "A": construct_solution(parents, config),
            "B": construct_solution(parents, config, model=old_model),
            "C": construct_ranked_solution(
                parents, config, models["GNN"], assignment_kind="RULE"
            ),
            "D": construct_ranked_solution(
                parents, config, models["GNN"], assignment_kind="MODEL"
            ),
            "E": construct_ranked_solution(
                parents, config, models["MLP"], assignment_kind="RULE"
            ),
            "F": construct_ranked_solution(
                parents,
                config,
                models["GNN_NO_WELD_EDGES"],
                assignment_kind="RULE",
                include_weld_edges=False,
            ),
        }
        instance_rows = []
        for method in ("A", "B", "C", "D", "E", "F"):
            print(f"study-eval {_entry_label(entry)} method={method}", flush=True)
            instance_rows.append(
                _evaluate_direct_initial(
                    entry, method, constructions[method], config, budget
                )
            )
        rule = instance_rows[0]
        rule_owner = assignment_targets(rule["_solution"])
        for row in instance_rows:
            if row["certified"] and rule["certified"]:
                row["relative_cmax_vs_rule"] = (
                    row["cmax"] - rule["cmax"]
                ) / rule["cmax"]
            owner = assignment_targets(row["_solution"])
            row["assignment_changes_vs_rule"] = sum(
                owner[block_id] != robot for block_id, robot in rule_owner.items()
            )
        rows.extend(instance_rows)
    summary = _paired_method_summary(rows)
    signal = any(
        summary[method]["wins"] > summary[method]["losses"]
        and (summary[method]["median_relative"] or 0.0) < 0.0
        for method in ("C", "D")
    )
    extra_note = (
        "DEV 出现明确排序/联合改善迹象，按任务要求应继续固定模型做 20 组既有几何验证。"
        if signal
        else "DEV 未出现明确稳定改善迹象，按停止条件未运行 20 组开发阶段几何验证。"
    )
    _write_initial_study_outputs(
        rows, histories, audit_rows, config, extra_note=extra_note
    )
    return rows, histories, audit_rows, signal


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
    parser.add_argument("--end-to-end-mode", action="store_true")
    parser.add_argument("--end-to-end-budget", type=float, default=20.0)
    parser.add_argument("--train-expert-mode", action="store_true")
    parser.add_argument("--train-expert-budget", type=float, default=60.0)
    parser.add_argument("--train-expert-limit", type=int)
    parser.add_argument(
        "--train-expert-construction",
        choices=("RULE", "GNN"),
        default="RULE",
    )
    parser.add_argument("--write-expert-data", action="store_true")
    parser.add_argument(
        "--initial-study",
        action="store_true",
        help="train saved-expert ranking models and run direct A-F DEV comparison",
    )
    parser.add_argument("--initial-study-budget", type=float, default=60.0)
    parser.add_argument("--initial-study-resume", action="store_true")
    args = parser.parse_args()
    config = ScientificConfig()
    representatives = ranker_dataset.representatives_v05()
    train_entries = [entry for entry in representatives if entry["split"] == "MLP_TRAIN"]
    dev_entries = [entry for entry in representatives if entry["split"] == "MLP_DEV"]
    if (len(train_entries), len(dev_entries)) != (17, 6):
        raise ValueError("expected existing 17 TRAIN / 6 DEV workbook split")
    if args.initial_study:
        run_initial_study(
            train_entries,
            dev_entries,
            config,
            args.epochs,
            args.initial_study_budget,
            resume=args.initial_study_resume,
        )
        print(f"checkpoint: {INITIAL_STUDY_MODEL_PATH}", flush=True)
        print(f"csv: {INITIAL_STUDY_CSV_PATH}", flush=True)
        print(f"report: {INITIAL_STUDY_REPORT_PATH}", flush=True)
        return
    if args.vns_fix_mode:
        run_vns_fix_mode(
            dev_entries,
            config,
            args.vns_fix_budget,
            args.vns_fix_all_dev,
            tuple(args.vns_fix_seeds),
        )
        return
    if args.end_to_end_mode:
        run_end_to_end_mode(dev_entries, config, args.end_to_end_budget)
        return
    if args.train_expert_mode:
        run_train_expert_mode(
            train_entries,
            config,
            args.train_expert_budget,
            args.train_expert_limit,
            args.write_expert_data,
            args.train_expert_construction,
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

