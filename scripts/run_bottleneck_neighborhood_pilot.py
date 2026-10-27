"""Fixed-pattern offline opportunity check; no C2/C4 candidate screening."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta_baselines.common import BaselineStatus, CommonBaselineEvaluator
from mrta_ranker import dataset
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import OperationKind, ScientificConfig, SplitKind
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map
from mrta_search.initialization import (
    InitializationStrategy, _construct, _patterns, bounded_insertion_positions,
)
from mrta_search.gnn_constructor import construct_solution
from mrta_search.lns import (
    BottleneckSelection, DestroyOperator, RepairOperator,
    repair_partial_state, select_bottleneck_parents,
)

CSV_PATH = ROOT / "results/bottleneck_neighborhood_pilot_20261027.csv"
REPORT_PATH = ROOT / "docs/BOTTLENECK_NEIGHBORHOOD_PILOT_20261027.md"
NAMES = ("g21_w035", "g15_w035", "g36_w086", "g33_w055", "g31_w055", "g45_w082")


def route_key(solution):
    return tuple(route.block_ids for route in solution.routes)


def owners(solution):
    return {b: r for r, route in enumerate(solution.routes) for b in route.block_ids}


def loads(candidate, config):
    blocks = block_map(candidate.solution, config)
    return [sum(config.process_time(blocks[b].length) for b in route.block_ids)
            for route in candidate.solution.routes]


def change_summary(initial, candidate, config):
    old, new = owners(initial.solution), owners(candidate.solution)
    changes = [(b, old[b], new[b]) for b in old if old[b] != new[b]]
    # Relative order of retained common blocks, separate from membership changes.
    reordered = []
    for r in range(4):
        common = set(initial.solution.routes[r].block_ids) & set(candidate.solution.routes[r].block_ids)
        a = [b for b in initial.solution.routes[r].block_ids if b in common]
        b = [b for b in candidate.solution.routes[r].block_ids if b in common]
        if a != b:
            reordered.append(r)
    return {
        "changed_parents": sorted({b.split("::", 1)[0] for b, _, _ in changes}),
        "block_transfers": changes,
        "same_rail_transfers": sum(a // 2 == b // 2 for _, a, b in changes),
        "cross_rail_transfers": sum(a // 2 != b // 2 for _, a, b in changes),
        "changed_routes": [r for r in range(4) if route_key(initial.solution)[r] != route_key(candidate.solution)[r]],
        "reordered_routes": reordered,
        "max_process_before": max(loads(initial, config)),
        "max_process_after": max(loads(candidate, config)),
        "travel_before": initial.metrics.total_empty_travel,
        "travel_after": candidate.metrics.total_empty_travel,
        "wait_before": initial.metrics.total_waiting,
        "wait_after": candidate.metrics.total_waiting,
        "cmax_before": initial.metrics.cmax,
        "cmax_after": candidate.metrics.cmax,
    }


def opportunities(initial, partial, config, insertion_limit):
    blocks = block_map(initial.solution, config)
    patterns = {p.parent_id: p for p in initial.solution.patterns}
    old = owners(initial.solution)
    movable = same_rail = cross_rail = positions = 0
    eligible_counts = Counter()
    for b in blocks.values():
        if b.parent_id not in partial.removed_parent_ids:
            continue
        pattern = patterns[b.parent_id]
        legal = [r for r in range(4) if robot_is_eligible(b, r, config)]
        if pattern.kind is SplitKind.X_SPLIT:
            legal = [(0 if pattern.rail.value == "UPPER" else 2) + int(b.block_id.endswith("::1"))]
        alternatives = [r for r in legal if r != old[b.block_id]]
        movable += bool(alternatives)
        same_rail += any(r // 2 == old[b.block_id] // 2 for r in alternatives)
        cross_rail += any(r // 2 != old[b.block_id] // 2 for r in alternatives)
        positions += sum(len(bounded_insertion_positions(len(partial.routes[r]), insertion_limit)) for r in legal)
        eligible_counts[len(legal)] += 1
    return movable, same_rail, cross_rail, positions, dict(eligible_counts)


def initial_candidate(parents, config, evaluator):
    started = time.perf_counter()
    # Directly use the existing REGION constructor, then the same public
    # evaluation as every completed neighbour. Fall back only on real failure.
    solution = _construct(parents, _patterns(parents, config), config, insertion_limit=8,
                          strategy=InitializationStrategy.REGION_COOPERATIVE_INITIALIZATION)
    initial = evaluator.evaluate(solution, source="REGION")
    if initial.status is not BaselineStatus.COMPLETED:
        print(f"REGION fallback required: {initial.status.value}", flush=True)
        result = construct_solution(parents, config)
        initial = evaluator.evaluate(result.solution, source="FALLBACK:RULE_X")
    if initial.status is not BaselineStatus.COMPLETED:
        raise RuntimeError(f"Common initialization failed: {initial.status}")
    return initial, time.perf_counter() - started


def run_group(entry, initial, evaluator, selection, q, args, config):
    started = time.perf_counter()
    row = dict(instance=entry["sheet_name"], workbook=entry["relative_path"], selection=selection.value,
               q=q, initial_cmax=initial.metrics.cmax, initial_source=initial.source,
               attempts=0, unique_partials=0, constructed_attempts=0, unique_candidates=0,
               direction_feasible=0, certified=0, improved=0, best_cmax=None, best_improvement_pct=None,
               restored_original=0, changed_assignment=0, same_rail_candidates=0,
               cross_rail_candidates=0, reordered_multiple_routes=0, changed_multiple_routes=0,
               insertion_evaluations=0, movable_blocks=0, same_rail_movable_blocks=0,
               cross_rail_movable_blocks=0, legal_positions=0,
               insertion_limit=args.insertion_limit, repair_budget=args.repair_budget, seed=args.seed,
               applicability="APPLICABLE")
    partials, candidates = set(), set()
    statuses, rejections = Counter(), Counter()
    best = None
    example = None
    for attempt in range(args.attempts):
        # Same attempt seeds across selectors; deterministic per instance/q.
        seed = args.seed + NAMES.index(entry["sheet_name"]) * 10000 + q * 100 + attempt
        partial = select_bottleneck_parents(initial.solution, initial.directions, initial.schedule,
                                            selection, q=q, seed=seed, config=config)
        if partial is None:
            row["applicability"] = "NO_WAIT_NOT_APPLICABLE"
            break
        row["attempts"] += 1
        partials.add(partial.removed_parent_ids)
        movable, same, cross, positions, eligible = opportunities(initial, partial, config, args.insertion_limit)
        row["movable_blocks"] += movable
        row["same_rail_movable_blocks"] += same
        row["cross_rail_movable_blocks"] += cross
        row["legal_positions"] += positions
        repaired = repair_partial_state(initial.solution, initial.directions, partial,
                                        DestroyOperator.RANDOM_REMOVAL, RepairOperator.REGRET_2_REPAIR,
                                        config, insertion_limit=args.insertion_limit,
                                        max_insertion_evaluations=args.repair_budget,
                                        preserve_patterns=True)
        row["insertion_evaluations"] += repaired.insertion_evaluations
        if repaired.candidate is None:
            rejections[repaired.rejection_reason] += 1
            continue
        row["constructed_attempts"] += 1
        solution = repaired.candidate.solution
        # Fixed blocks/patterns and untouched route skeleton are checked here;
        # final geometry/safety certification remains the existing public path.
        assert solution.patterns == initial.solution.patterns
        assert set(block_map(solution, config)) == set(block_map(initial.solution, config))
        for r in range(4):
            retained = tuple(b for b in solution.routes[r].block_ids
                             if b.split("::", 1)[0] not in partial.removed_parent_ids)
            assert retained == partial.routes[r]
        key = route_key(solution)
        row["restored_original"] += key == route_key(initial.solution)
        if key in candidates:
            continue
        candidates.add(key)
        evaluated = evaluator.evaluate(solution, source=f"{selection.value}:q{q}:{attempt}")
        statuses[evaluated.status.value] += 1
        row["direction_feasible"] += evaluated.schedule is not None
        if evaluated.status is not BaselineStatus.COMPLETED:
            if example is None:
                example = dict(removed=list(partial.removed_parent_ids), movable=movable,
                               legal_positions=positions, eligible_robot_counts=eligible,
                               status=evaluated.status.value, diagnostics=list(evaluated.diagnostics),
                               repair_trace=dict(repaired.candidate.provenance)["repair_trace"])
            continue
        row["certified"] += 1
        changed = change_summary(initial, evaluated, config)
        row["changed_assignment"] += bool(changed["block_transfers"])
        row["same_rail_candidates"] += bool(changed["same_rail_transfers"])
        row["cross_rail_candidates"] += bool(changed["cross_rail_transfers"])
        row["reordered_multiple_routes"] += len(changed["reordered_routes"]) > 1
        row["changed_multiple_routes"] += len(changed["changed_routes"]) > 1
        epsilon = 1e-9 * max(1.0, initial.metrics.cmax)
        row["improved"] += evaluated.metrics.cmax < initial.metrics.cmax - epsilon
        if best is None or evaluated.metrics.cmax < best.metrics.cmax:
            best = evaluated
        if example is None and evaluated.metrics.cmax >= initial.metrics.cmax - epsilon:
            example = dict(removed=list(partial.removed_parent_ids), movable=movable,
                           legal_positions=positions, eligible_robot_counts=eligible,
                           status=evaluated.status.value, changes=changed,
                           repair_trace=dict(repaired.candidate.provenance)["repair_trace"])
    row["unique_partials"] = len(partials)
    row["unique_candidates"] = len(candidates)
    if best is not None:
        row["best_cmax"] = best.metrics.cmax
        row["best_improvement_pct"] = 100 * (initial.metrics.cmax - best.metrics.cmax) / initial.metrics.cmax
    row["statuses"] = json.dumps(statuses, ensure_ascii=False, sort_keys=True)
    row["repair_rejections"] = json.dumps(rejections, ensure_ascii=False, sort_keys=True)
    row["runtime_s"] = time.perf_counter() - started
    print(f"{row['instance']} {selection.value} q={q}: "
          f"unique={len(candidates)} direction={row['direction_feasible']} "
          f"certified={row['certified']} improved={row['improved']} "
          f"best={row['best_cmax']} time={row['runtime_s']:.3f}s", flush=True)
    return row, best, example


def short(block_id):
    return block_id.split("|", 1)[0] + ":" + block_id.rsplit("row=", 1)[-1]


def timeline(candidate):
    # Plain text timeline needs no image/file or visualization framework.
    width = 100
    chars = {OperationKind.MOVE: "m", OperationKind.SETUP: "p", OperationKind.WELD: "W",
             OperationKind.POST: "p", OperationKind.WAIT: "."}
    lines = [f"0{' ' * 88}{candidate.metrics.cmax:.1f}s"]
    for r in range(4):
        line = [" "] * width
        for op in candidate.schedule.operations:
            if op.robot_id != r:
                continue
            start = min(width - 1, int(op.start_time / candidate.metrics.cmax * width))
            end = min(width, max(start + 1, math.ceil(op.end_time / candidate.metrics.cmax * width)))
            line[start:end] = [chars[op.kind]] * (end - start)
        lines.append(f"R{r} |{''.join(line)}| {candidate.schedule.robot_completion[r]:.1f}")
    return "\n".join(lines)


def save(rows, cases, examples, initial_records, args):
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 瓶颈驱动固定模式邻域开发实验（2026-10-27）", "",
             f"seed={args.seed}；q={args.q}；每组 {args.attempts} 次；REGRET_2_REPAIR；"
             f"insertion_limit={args.insertion_limit}；max_insertion_evaluations={args.repair_budget}。",
             "FORMAL_SCOPE_V2：焊速 0.0108、空走 0.20 m/s，准备/后处理 20/30 s，X/Y 窗口 0.50 m，最短子段 0.20 m。开放路线、首任务免费部署。",
             "共同 REGION 初解经公共方向 DP、B32 和原 certifier；仅初解失败才使用已有 construct_solution 的 RULE/X。"
             "每组均从同一个不可变初解开始；候选逐个公共评价，无 C2/C4。模式规范化改变时统一拒绝。",
             "WAIT_ANCHORED 是真实 WAIT 位置、时段和端点空间相关代理，未声称识别真实 blocker；调度器未改动。"
             "无 WAIT 的组不适用。DEADLOCK 只表示 B32 有限搜索失败；失败 Cmax 留空，不参与均值。",
             "唯一候选按四条块路线去重（方向随后由公共 DP 决定）；组间独立评价。restored_original 按全部成功构造次数计数。"
             "方向可行数按实际进入 B32 的候选计；严格改善容差为 1e-9×初始 Cmax。运行秒为选区、修复和公共评价总墙钟，不含初解。",
             "", "## 实例与共同初解", ""]
    for entry, initial, elapsed in initial_records:
        lines.append(f"- {entry['sheet_name']}：`{entry['relative_path']}`；{len(initial.solution.parents)} 父焊缝；"
                     f"{initial.source}；Cmax={initial.metrics.cmax:.6f}s；初解耗时 {elapsed:.3f}s；"
                     f"模式 {dict(Counter(p.kind.value for p in initial.solution.patterns))}；"
                     f"加工负载 {[round(v, 3) for v in loads(initial, ScientificConfig())]}；"
                     f"WAIT={initial.metrics.total_waiting:.3f}s。")
    lines += ["", "数据只通过现有 representatives_v05 的 MLP_DEV 六实例和 load_parents 白名单读取。"
              "g15 的历史目录名为 ID_TEST，但它属于已消耗开发白名单；未读取当前封存测试角色的几何。", "",
              "## 全部分组结果", "",
              "|实例|选区|q|初始 Cmax|尝试/唯一部分解|唯一候选|方向可行|认证|改善|最佳 Cmax|最佳改善%|秒|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        best = "—" if row["best_cmax"] is None else f"{row['best_cmax']:.3f}"
        pct = "—" if row["best_improvement_pct"] is None else f"{row['best_improvement_pct']:.3f}"
        lines.append(f"|{row['instance']}|{row['selection']}|{row['q']}|{row['initial_cmax']:.3f}|"
                     f"{row['attempts']}/{row['unique_partials']}|{row['unique_candidates']}|{row['direction_feasible']}|"
                     f"{row['certified']}|{row['improved']}|{best}|{pct}|{row['runtime_s']:.2f}|")
    lines += ["", "## 各组选区的最佳相对改善候选", "",
              "ID 简写保留 inst、row 编号和子块后缀。下面给出每类选区最佳相对改善，而非把不同实例的 Cmax 直接比较。", ""]
    showcase = None
    for selection, (entry, initial, best) in cases.items():
        summary = change_summary(initial, best, ScientificConfig())
        transfers = "; ".join(f"`{b}` R{a}→R{r}" for b, a, r in summary["block_transfers"]) or "无归属变化"
        lines += [f"### {selection} / {entry['sheet_name']} / {best.source}", "",
                  f"归属：{transfers}。改变路线 {summary['changed_routes']}；共同保留块相对顺序改变路线 {summary['reordered_routes']}。",
                  f"最大加工负载 {summary['max_process_before']:.3f}→{summary['max_process_after']:.3f}s；"
                  f"总空走 {summary['travel_before']:.3f}→{summary['travel_after']:.3f}s；"
                  f"总 WAIT {summary['wait_before']:.3f}→{summary['wait_after']:.3f}s；"
                  f"Cmax {summary['cmax_before']:.3f}→{summary['cmax_after']:.3f}s。", ""]
        if summary["block_transfers"] and len(summary["reordered_routes"]) > 1:
            pct = (initial.metrics.cmax - best.metrics.cmax) / initial.metrics.cmax
            if showcase is None or pct > showcase[0]:
                showcase = (pct, entry, initial, best)
    if showcase is not None:
        _, entry, initial, best = showcase
        lines += [f"## 联合改善案例路线和时间线：{entry['sheet_name']} / {best.source}", "",
                  "任务标记为 inst:row::子块/方向。W=焊接，p=准备/后处理，m=空走，.=WAIT；"
                  "时间线按各自 Cmax 缩放到 100 列，仅供解释。", ""]
        for r in range(4):
            before = " → ".join(f"{short(b)}/{d}" for b, d in zip(initial.solution.routes[r].block_ids, initial.directions[r]))
            after = " → ".join(f"{short(b)}/{d}" for b, d in zip(best.solution.routes[r].block_ids, best.directions[r]))
            lines.append(f"- R{r}：`{before}` → `{after}`")
        lines += ["", "```text", "BEFORE", timeline(initial), "AFTER", timeline(best), "```", ""]
    lines += ["## 构造与失败诊断", "",
              "movable/positions 是每次部分解的静态合法机会累计（位置数不含两方向和双子块笛卡尔组合），不是已经枚举的动作数。"
              "CSV 另含真实插入评分次数、固定模式拒绝、原结构恢复、同轨/跨轨迁移和多路线重排计数。", ""]
    for selection in BottleneckSelection:
        subset = [r for r in rows if r["selection"] == selection.value]
        fields = ("attempts", "unique_partials", "unique_candidates", "certified", "improved", "restored_original",
                  "movable_blocks", "same_rail_movable_blocks", "cross_rail_movable_blocks", "legal_positions",
                  "changed_assignment", "same_rail_candidates", "cross_rail_candidates", "reordered_multiple_routes")
        lines += [f"- {selection.value}：" + json.dumps({k: sum(r[k] for r in subset) for k in fields}, ensure_ascii=False)]
    lines += ["", "少量实际未改善/失败案例（插入标记为 block→R@位置/方向）：", ""]
    for label, example in list(examples.items())[:4]:
        trace = "; ".join(f"{short(b)}→R{r}@{k}/{h}" for _, moves in example["repair_trace"] for b, r, k, h in moves)
        lines += [f"- {label}：{example['status']}；移除 {[short(p) for p in example['removed']]}；"
                  f"可迁移块 {example['movable']}，静态合法位置 {example['legal_positions']}，"
                  f"合法机器人数量分布 {example['eligible_robot_counts']}；实际插入 `{trace}`。"]
        if "changes" in example:
            c = example["changes"]
            lines.append(f"  Cmax {c['cmax_before']:.3f}→{c['cmax_after']:.3f}，"
                         f"改变归属父数 {len(c['changed_parents'])}，重排路线 {c['reordered_routes']}。")
    statuses = sum((Counter(json.loads(r["statuses"])) for r in rows), Counter())
    restored = sum(r["restored_original"] for r in rows)
    attempted = sum(r["attempts"] for r in rows)
    hard_failures = [r for r in rows if r["direction_feasible"] > 0 and r["certified"] == 0
                     and json.loads(r["statuses"]).get("DEADLOCK", 0) == r["direction_feasible"]]
    zero_instances = []
    for name in dict.fromkeys(r["instance"] for r in rows):
        subset = [r for r in rows if r["instance"] == name]
        if not sum(r["improved"] for r in subset):
            zero_instances.append((name, sum(r["attempts"] for r in subset),
                                   sum(r["unique_candidates"] for r in subset)))
    coverage = {s.value: len({r["instance"] for r in rows if r["selection"] == s.value and r["improved"]})
                for s in BottleneckSelection}
    lines += ["", "## 结论", "",
              f"本轮完成 {attempted} 次适用选区尝试、{sum(r['unique_candidates'] for r in rows)} 个组内唯一完整候选；"
              f"状态 {dict(statuses)}；{sum(r['improved'] for r in rows)} 个组内改善候选。跨组可能重合，不能据此声称全局同样数量的独特专家。"
              f"选区与候选评价总耗时 {sum(r['runtime_s'] for r in rows):.3f}s（不含初解与输入加载）。",
              f"原结构仅恢复 {restored}/{sum(r['constructed_attempts'] for r in rows)} 次；"
              f"累计组内唯一部分解 {sum(r['unique_partials'] for r in rows)}，"
              "且实际出现同轨左右迁移、跨轨迁移和多路线相对重排。因此当前没有证据把失败主要归因于选区重复或机器人完全不可迁移。",
              "Regret-2 的 `_parent_alternatives` 依次比较最大加工负载和局部有向空走增量，没有 WAIT/同轨路线交错项；"
              "它的两优方案可能只是同一归属/位置的不同方向提示，方向随后又被公共 DP 重算。"
              "`remaining_budget` 按确定性机器人/位置/方向循环截断，双子块时可能只访问动作前缀；静态合法位置数不等于全部已搜索。"
              "这些是候选质量限制的代码机制，尚未做评分/枚举消融，不能把它们当成已证实因果。此次未改 Regret-2 核心评分。",
              "方向可行而全部 B32 失败的分组：" + "; ".join(
                  f"{r['instance']}/{r['selection']}/q{r['q']}，{r['direction_feasible']} 个" for r in hard_failures) + "。",
              "本轮无改善实例（实例、尝试、完整唯一候选）：" + str(zero_instances) + "。"
              "请结合静态合法位置、迁移资格及真实修复 trace 判断，不能把无改善解释为数学不可改善。",
              "各类选区改善覆盖实例数：" + str(coverage) + "。"
              "当前结果支持先研究共同修复器的联合补全和调度感知能力，并保持 RANDOM 对照。"
              "WAIT 的适用样本少，应按存在等待的共同实例比较，不能直接与全部六例比较成功率。",
              "已有机会证据，尚无足够广度和独特专家数量直接支持启动神经训练。先在相同移除集合上改善 Regret-2 的"
              "有效归属/位置方案多样性和可调度率，再建立部分解补全专家。真正 blocker 证据值得在后续小范围旁路保留，"
              "用于解释 WAIT 相关重构；本轮的代理结果不构成 blocker 因果证明。",
              ("本轮实际初解只出现 WHOLE/Y_SPLIT，未实测 X_SPLIT 邻域收益；X 固定左右约束继续复用已有代码和相关测试。"
               if not any(p.kind is SplitKind.X_SPLIT for _, initial, _ in initial_records for p in initial.solution.patterns)
               else "X_SPLIT 固定左右约束继续复用已有代码和相关测试。"),
              "", "## 复现", "", "```powershell",
              f"& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_bottleneck_neighborhood_pilot.py "
              f"--instances {' '.join(args.instances)} --attempts {args.attempts} --q {' '.join(map(str, args.q))} "
              f"--seed {args.seed} --insertion-limit {args.insertion_limit} --repair-budget {args.repair_budget}",
              "```", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--q", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--insertion-limit", type=int, default=8)
    parser.add_argument("--repair-budget", type=int, default=4096)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("attempts must be positive")
    dev = {e["sheet_name"]: e for e in dataset.representatives_v05() if e["split"] == "MLP_DEV"}
    config = ScientificConfig()
    rows, cases, examples, initial_records = [], {}, {}, []
    for name in args.instances:
        entry = dev[name]
        parents = dataset.load_parents(entry)
        evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V2, time_limit=86400.0)
        initial, elapsed = initial_candidate(parents, config, evaluator)
        initial_records.append((entry, initial, elapsed))
        print(f"INITIAL {name} {initial.source} Cmax={initial.metrics.cmax:.6f} time={elapsed:.3f}", flush=True)
        for q in args.q:
            for selection in BottleneckSelection:
                row, best, example = run_group(entry, initial, evaluator, selection, q, args, config)
                rows.append(row)
                if best is not None:
                    pct = (initial.metrics.cmax - best.metrics.cmax) / initial.metrics.cmax
                    previous = cases.get(selection.value)
                    if previous is None or pct > (previous[1].metrics.cmax - previous[2].metrics.cmax) / previous[1].metrics.cmax:
                        cases[selection.value] = (entry, initial, best)
                if example is not None and selection.value not in examples:
                    examples[selection.value] = example
        # Keep this invocation's completed instances in the two approved outputs.
        save(rows, cases, examples, initial_records, args)
    print(f"Saved {CSV_PATH}\nSaved {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
