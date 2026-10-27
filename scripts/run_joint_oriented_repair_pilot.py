"""DEV-only A0/A1 and bottleneck pilot with explicit insertion directions."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrta_ranker import dataset
from mrta_reference.geometry import oriented_endpoints, robot_is_eligible, build_legal_pattern_catalog
from mrta_reference.model import OperationKind, ParentWeld, Route, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.solution import block_map, canonicalize
from mrta_search.direction import _initial_error
from mrta_search.initialization import _patterns, _construct_region_cooperative_initialization
from mrta_search.gnn_constructor import construct_solution
from mrta_search.lns import (
    BottleneckSelection, oriented_route_state, oriented_insert_variants,
    repair_partial_state_oriented, select_bottleneck_parents,
)

NAMES = ("g21_w035", "g15_w035", "g36_w086", "g33_w055", "g31_w055", "g45_w082")
CSV_PATH = ROOT / "results/joint_oriented_repair_pilot.csv"
REPORT_PATH = ROOT / "docs/JOINT_ORIENTED_REPAIR_PILOT.md"
METHODS = ("A0_RANDOM", "A1_RANDOM", "A1_LOAD", "A1_TRAVEL", "A1_WAIT_ANCHORED")


@dataclass(frozen=True)
class DirectedSolution:
    solution: object
    directions: tuple
    schedule: object


def schedule_explicit(solution, directions, config):
    """Basic input/output checks and exactly one call to the existing B32."""
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    normalized = canonicalize(solution.parents, solution.patterns, solution.routes, config, scope=FORMAL_SCOPE_V2)
    if normalized.patterns != solution.patterns or normalized.routes != solution.routes:
        raise ValueError("fixed patterns or blocks changed during canonicalization")
    if len(directions) != 4:
        raise ValueError("four explicit direction vectors, including empty routes, required")
    actual = [b for route in solution.routes for b in route.block_ids]
    if Counter(actual) != Counter(blocks.keys()):
        raise ValueError("block coverage/uniqueness mismatch")
    for r, route in enumerate(solution.routes):
        if len(route.block_ids) != len(directions[r]) or any(d not in (0, 1) for d in directions[r]):
            raise ValueError("invalid direction vector")
        if any(not robot_is_eligible(blocks[b], r, config) for b in route.block_ids):
            raise ValueError("robot eligibility mismatch")
    explicit = {r: tuple(directions[r]) for r in range(4)}
    schedule = reference_schedule_formal(solution, config, scope=FORMAL_SCOPE_V2, orientations=explicit)
    if schedule.status is ScheduleStatus.FEASIBLE:
        welds = [op for op in schedule.operations if op.kind is OperationKind.WELD]
        if Counter(op.block_id for op in welds) != Counter(blocks.keys()):
            raise ValueError("FEASIBLE schedule omitted or repeated a welding block")
        if schedule.cmax is None or not math.isfinite(schedule.cmax):
            raise ValueError("FEASIBLE schedule has no finite Cmax")
        if schedule.directions != tuple(explicit[r] for r in range(4)):
            raise ValueError("scheduler did not preserve explicit directions")
        requested = {b: (r, d) for r, route in enumerate(solution.routes)
                     for b, d in zip(route.block_ids, directions[r])}
        for op in welds:
            r, d = requested[op.block_id]
            start, end = oriented_endpoints(blocks[op.block_id], d)
            if op.robot_id != r or math.dist(op.start, start) > config.numeric_epsilon or math.dist(op.end, end) > config.numeric_epsilon:
                raise ValueError("welding operation does not match directed assignment")
    return DirectedSolution(solution, tuple(tuple(v) for v in directions), schedule)


def greedy_route_cost(route, blocks, config):
    """REGION's optional cost hook: sequential local insertion, never route DP."""
    state = oriented_route_state(((), (), (), ()), ((), (), (), ()), blocks, config)
    for b in route:
        robot = 0 if robot_is_eligible(blocks[b], 0, config) else 2
        rows, _ = oriented_insert_variants(state, blocks[b], robot, len(state.routes[robot]), blocks, config, keep=1)
        # A route used by REGION lies entirely on one eligible rail.
        if not rows:
            raise ValueError("no local directed route construction")
        state = rows[0][0]
    return state.travel


def build_directions_by_insertion(solution, config, first_bits):
    blocks = block_map(solution, config)
    active = [r for r, route in enumerate(solution.routes) if route.block_ids]
    locked = {solution.routes[r].block_ids[0]: d for r, d in zip(active, first_bits)}
    points = {r: oriented_endpoints(blocks[solution.routes[r].block_ids[0]], locked[solution.routes[r].block_ids[0]])[0]
              for r in active}
    if _initial_error(points, config) is not None:
        return None
    state = oriented_route_state(((), (), (), ()), ((), (), (), ()), blocks, config)
    # First tasks are deployed before route extensions; each append chooses
    # its own direction and may revise only the immediate predecessor.
    for depth in range(max(map(lambda r: len(r.block_ids), solution.routes), default=0)):
        for r in active:
            route = solution.routes[r].block_ids
            if depth >= len(route):
                continue
            rows, _ = oriented_insert_variants(state, blocks[route[depth]], r, len(state.routes[r]), blocks, config,
                                               locked_directions=locked, keep=1)
            if not rows:
                return None
            state = rows[0][0]
    return state.directions


def common_initial(parents, config):
    attempts = []
    # The hook and improve_routes=False bypass both hidden REGION route-DP
    # proxies and its old route toolkit. Assignment/boundary logic is reused.
    for scheme in ("REGION_LOCAL", "RULE_X_LOCAL"):
        try:
            solution = (_construct_region_cooperative_initialization(
                parents, _patterns(parents, config), config,
                route_cost=greedy_route_cost, improve_routes=False,
            ) if scheme == "REGION_LOCAL" else construct_solution(parents, config).solution)
        except (ValueError, ArithmeticError) as error:
            attempts.append(dict(scheme=scheme, status="CONSTRUCTION_FAILED", diagnostics=(str(error),)))
            continue
        active = sum(bool(r.block_ids) for r in solution.routes)
        for bits in itertools.product((0, 1), repeat=active):
            directions = build_directions_by_insertion(solution, config, bits)
            if directions is None:
                attempts.append(dict(scheme=scheme, first_bits=bits, status="ILLEGAL_INITIAL_POINTS"))
                continue
            candidate = schedule_explicit(solution, directions, config)
            attempts.append(dict(scheme=scheme, first_bits=bits, status=candidate.schedule.status.value,
                                 diagnostics=candidate.schedule.diagnostics))
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                return candidate, attempts
    return None, attempts


def key(candidate):
    return tuple(r.block_ids for r in candidate.solution.routes), candidate.directions


def summary(initial, candidate, config):
    def info(c):
        blocks = block_map(c.solution, config)
        owner = {b: r for r, route in enumerate(c.solution.routes) for b in route.block_ids}
        bits = {b: d for route, ds in zip(c.solution.routes, c.directions) for b, d in zip(route.block_ids, ds)}
        state = oriented_route_state(tuple(r.block_ids for r in c.solution.routes), c.directions, blocks, config)
        return owner, bits, state
    old, old_bits, before = info(initial)
    new, new_bits, after = info(candidate)
    transfers = [(b, old[b], new[b]) for b in old if old[b] != new[b]]
    flips = [(b, old_bits[b], new_bits[b]) for b in old if old_bits[b] != new_bits[b]]
    reordered = []
    for r in range(4):
        common = set(before.routes[r]) & set(after.routes[r])
        if [b for b in before.routes[r] if b in common] != [b for b in after.routes[r] if b in common]:
            reordered.append(r)
    return dict(transfers=transfers, flips=flips, reordered_routes=reordered,
                process_before=before.loads, process_after=after.loads,
                travel_before=before.travel, travel_after=after.travel,
                wait_before=sum(op.duration for op in initial.schedule.operations if op.kind is OperationKind.WAIT),
                wait_after=(sum(op.duration for op in candidate.schedule.operations if op.kind is OperationKind.WAIT)
                            if candidate.schedule.status is ScheduleStatus.FEASIBLE else None),
                cmax_before=initial.schedule.cmax, cmax_after=candidate.schedule.cmax)


def run_group(name, method, q, partials, initial, args, config):
    started = time.perf_counter()
    row = dict(instance=name, method=method, q=q, initial_cmax=initial.schedule.cmax if initial else None,
               attempts=0, unique_partials=0, constructed=0, unique_candidates=0, schedule_success=0,
               improved=0, best_cmax=None, improvement_pct=None, runtime_s=0,
               restored_original=0, changed_assignment=0, changed_direction=0, reordered=0,
               joint_expert_candidates=0, direction_order_only_improved=0, insertion_evaluations=0,
               movable_blocks=0, removed_from_critical_route=0,
               insertion_limit=args.insertion_limit, repair_budget=args.repair_budget, seed=args.seed,
               statuses="{}", repair_failures="{}", applicability="APPLICABLE")
    if initial is None or not partials:
        row["applicability"] = "INITIALIZATION_FAILED" if initial is None else "NO_WAIT_NOT_APPLICABLE"
        return row, None, {}, None
    unique, removed_sets, outcomes, attempt_keys = set(), set(), {}, {}
    statuses, failures = Counter(), Counter()
    best = None
    example = None
    blocks = block_map(initial.solution, config)
    critical = max(range(4), key=lambda r: initial.schedule.robot_completion[r])
    old_owners = {b: r for r, route in enumerate(initial.solution.routes) for b in route.block_ids}
    for attempt, partial in enumerate(partials):
        row["attempts"] += 1
        removed_sets.add(partial.removed_parent_ids)
        row["movable_blocks"] += sum(any(r != old_owners[b] and robot_is_eligible(blocks[b], r, config) for r in range(4))
                                     for b in blocks if blocks[b].parent_id in partial.removed_parent_ids)
        row["removed_from_critical_route"] += sum(blocks[b].parent_id in partial.removed_parent_ids
                                                 for b in initial.solution.routes[critical].block_ids)
        seed = args.seed + NAMES.index(name) * 10000 + q * 100 + attempt
        result = repair_partial_state_oriented(initial.solution, initial.directions, partial, config,
                                               fixed_directions=method.startswith("A0"), seed=seed,
                                               insertion_limit=args.insertion_limit, max_insertion_evaluations=args.repair_budget)
        row["insertion_evaluations"] += result.insertion_evaluations
        if result.candidate is None:
            failures[result.rejection_reason] += 1
            outcomes[attempt] = ("CONSTRUCTION_FAILED", None)
            continue
        row["constructed"] += 1
        solution = result.candidate.solution
        assert solution.patterns == initial.solution.patterns
        for r in range(4):
            assert tuple(b for b in solution.routes[r].block_ids if blocks[b].parent_id not in partial.removed_parent_ids) == partial.routes[r]
        if method.startswith("A0"):
            initial_bits = {b: d for route, ds in zip(initial.solution.routes, initial.directions) for b, d in zip(route.block_ids, ds)}
            assert all(initial_bits[b] == d for route, ds in zip(solution.routes, result.directions) for b, d in zip(route.block_ids, ds))
        candidate_key = (tuple(r.block_ids for r in solution.routes), result.directions)
        row["restored_original"] += candidate_key == key(initial)
        if candidate_key in unique:
            outcomes[attempt] = outcomes[next(i for i, k in attempt_keys.items() if k == candidate_key)]
            continue
        unique.add(candidate_key)
        attempt_keys[attempt] = candidate_key
        candidate = schedule_explicit(solution, result.directions, config)
        status = candidate.schedule.status
        statuses[status.value] += 1
        outcomes[attempt] = (status.value, candidate.schedule.cmax if status is ScheduleStatus.FEASIBLE else None)
        details = dict(instance=name, method=method, q=q, attempt=attempt, seed=seed,
                       removed=list(partial.removed_parent_ids), actions=[asdict(a) for a in result.actions],
                       changes=summary(initial, candidate, config), status=status.value,
                       diagnostics=candidate.schedule.diagnostics,
                       before_routes=[list(r.block_ids) for r in initial.solution.routes], before_directions=initial.directions,
                       after_routes=[list(r.block_ids) for r in solution.routes], after_directions=result.directions)
        if status is not ScheduleStatus.FEASIBLE:
            if example is None:
                example = details
            continue
        row["schedule_success"] += 1
        changes = details["changes"]
        row["changed_assignment"] += bool(changes["transfers"])
        row["changed_direction"] += bool(changes["flips"])
        row["reordered"] += bool(changes["reordered_routes"])
        if row["best_cmax"] is None or candidate.schedule.cmax < row["best_cmax"]:
            row["best_cmax"] = candidate.schedule.cmax
        improved = candidate.schedule.cmax < initial.schedule.cmax - 1e-9 * max(1, initial.schedule.cmax)
        if improved:
            row["improved"] += 1
            row["joint_expert_candidates"] += bool(changes["transfers"] and changes["flips"] and changes["reordered_routes"])
            row["direction_order_only_improved"] += bool(not changes["transfers"] and changes["flips"] and changes["reordered_routes"])
            if best is None or candidate.schedule.cmax < best["changes"]["cmax_after"]:
                best = details
        elif example is None:
            example = details
    row["unique_partials"] = len(removed_sets)
    row["unique_candidates"] = len(unique)
    row["statuses"] = json.dumps(statuses, sort_keys=True)
    row["repair_failures"] = json.dumps(failures, sort_keys=True)
    if row["best_cmax"] is not None:
        row["improvement_pct"] = 100 * (initial.schedule.cmax - row["best_cmax"]) / initial.schedule.cmax
    row["runtime_s"] = time.perf_counter() - started
    print(f"{name} {method} q{q}: candidates={len(unique)} success={row['schedule_success']} improved={row['improved']} best={row['best_cmax']} time={row['runtime_s']:.2f}s", flush=True)
    return row, best, outcomes, example


def x_split_smoke(config):
    parent = ParentWeld("formal-x", (0.0, 8.0), (4.0, 8.0))
    pattern = next(p for p in build_legal_pattern_catalog((parent,), config, FORMAL_SCOPE_V2)[parent.parent_id]
                   if p.kind is SplitKind.X_SPLIT and p.point_id == "BX_CENTER")
    solution = canonicalize((parent,), (pattern,), {0: ("formal-x::0",), 1: ("formal-x::1",)}, config, scope=FORMAL_SCOPE_V2)
    from mrta_search.lns import _remove_parent_set
    directions = build_directions_by_insertion(solution, config, (0, 1))
    initial = schedule_explicit(solution, directions, config)
    assert initial.schedule.status is ScheduleStatus.FEASIBLE
    result = repair_partial_state_oriented(solution, directions, _remove_parent_set(solution, (parent.parent_id,)),
                                           config, seed=20261027)
    if result.candidate is None:
        raise ValueError(f"X smoke repair failed: {result.rejection_reason}")
    candidate = schedule_explicit(result.candidate.solution, result.directions, config)
    assert candidate.solution.patterns == (pattern,)
    assert candidate.solution.routes[0].block_ids == ("formal-x::0",)
    assert candidate.solution.routes[1].block_ids == ("formal-x::1",)
    assert candidate.directions[2:] == ((), ())
    assert candidate.schedule.status is ScheduleStatus.FEASIBLE
    return dict(status=candidate.schedule.status.value, cmax=candidate.schedule.cmax,
                directions=candidate.directions, actions=[asdict(a) for a in result.actions])


def short(b):
    return b.split("|", 1)[0] + ":" + b.rsplit("row=", 1)[-1] if "row=" in b else b


def conclusion_lines(rows, paired):
    totals = {method: {k: sum(r[k] for r in rows if r['method'] == method)
                       for k in ('attempts', 'schedule_success', 'improved', 'joint_expert_candidates', 'runtime_s')}
              for method in METHODS}
    pairs = {k: sum(p[k] for p in paired) for k in ('both_success', 'a1_wins', 'ties', 'a0_wins', 'only_a1_success', 'only_a0_success')}
    a0, a1 = totals['A0_RANDOM'], totals['A1_RANDOM']
    joint = sum(totals[m]['joint_expert_candidates'] for m in METHODS if m.startswith('A1'))
    coverage = {m: len({r['instance'] for r in rows if r['method'] == m and r['improved']}) for m in METHODS}
    lines = [
        f"1. 方向嵌入的质量机会：相同 RANDOM 移除输入，A0/A1 改善候选 {a0['improved']}/{a1['improved']}，"
        f"组内唯一候选的 B32 成功数 {a0['schedule_success']}/{a1['schedule_success']}；"
        f"运行时间 {a0['runtime_s']:.3f}/{a1['runtime_s']:.3f}s。"
        f"逐次配对共同成功 {pairs['both_success']} 次，A1 胜/平/A0 胜为 {pairs['a1_wins']}/{pairs['ties']}/{pairs['a0_wins']}；"
        f"仅 A1/仅 A0 成功 {pairs['only_a1_success']}/{pairs['only_a0_success']}。"
        "配对按尝试计，重复候选复用该组已完成的调度结果，因此与去重成功数略有区别。",
        "2. 选区：改善覆盖实例数 " + str(coverage) + "。"
        "RANDOM 是必要对照；在本轮 LOAD/TRAVEL/WAIT 的比较中，应同时比较成功数、改善数和同实例最佳 Cmax。"
        "WAIT 不适用组不能计为失败，也不能用不同实例集合的原始成功率直接排名。",
        f"3. 联合专家：A1 中有 {joint} 个组内改善候选同时改变归属、共同块相对顺序及方向，"
        "已经具备完整动作及邻块翻转记录。该数未跨选区去重，不能当作独特训练样本数；只说明联合补全动作可执行。",
        "4. 下一步：已有方向嵌入和专家轨迹的开发证据，但当前可调度性、几何覆盖及初解质量仍不足以直接支持正式 GNN 训练结论。"
        "先用同一移除集合控制有效结构候选覆盖，改善局部评分对同轨交错/协调冲突的识别，并积累去重、多初解轨迹；"
        "避免把局部空走最优当作 Cmax 最优。本次不启动训练。",
        "新初解经逐步定向构造，和上一轮初解不同；不可用本轮改善数或相对改善直接宣称优于上一轮独立 DP 流程。",
    ]
    if a1['improved'] > a0['improved'] and pairs['a1_wins'] > pairs['a0_wins']:
        lines.insert(1, "本轮方向嵌入比固定方向有更多改善机会，但不是逐例占优。逐次配对的调度成功变化应单独判断，不能把改善候选增加解释为可调度率同步提高。")
    lines.insert(3, f"瓶颈选区的改善数：LOAD={totals['A1_LOAD']['improved']}、TRAVEL={totals['A1_TRAVEL']['improved']}、"
                 f"WAIT={totals['A1_WAIT_ANCHORED']['improved']}。LOAD/TRAVEL 与 RANDOM 使用同样尝试数，"
                 "本轮仍应保留 RANDOM；WAIT 应只在有等待的共同实例上解释。")
    for name in ('g33_w055', 'g36_w086', 'g45_w082'):
        group = [r for r in rows if r['instance'] == name and r['method'].startswith('A1')]
        if not group:
            continue
        success = sum(r['schedule_success'] for r in group)
        generated = sum(r['unique_candidates'] for r in group)
        moved = sum(r['changed_assignment'] for r in group)
        best = min((r['best_cmax'] for r in group if r['best_cmax'] is not None), default=None)
        lines.append(f"- {name}：A1 组内完整候选 {generated}，B32 成功 {success}，其中改变归属 {moved}；"
                     f"移除块中有其他合法机器人可选的累计次数 {sum(r['movable_blocks'] for r in group)}，"
                     f"来自当前最大完工路线的移除块累计 {sum(r['removed_from_critical_route'] for r in group)}；最佳 Cmax={best}。"
                     "这些计数不能把无改善解释为最优，或直接归因于完全缺乏合法动作。")
    return lines


def save(rows, initial_logs, cases, examples, paired, smoke, args):
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 方向嵌入联合插入修复开发实验", "",
             f"seed={args.seed}；q={args.q}；attempts={args.attempts}；insertion_limit={args.insertion_limit}；repair_budget={args.repair_budget}。",
             "FORMAL_SCOPE_V2，0.0108/0.20 m/s，准备/后处理 20/30s，X/Y 窗口 0.50m，最短子段 0.20m；开放路线、免费首部署。",
             "A0 固定每块初始方向；A1 枚举新块及前后邻块方向，最多保留两个局部配置。二者共享修复循环、不同结构 Regret-2、机器人均分预算和端点评分。"
             "A1_RANDOM 同时用于实验 A 和 B，只执行一遍。RANDOM 的 A0/A1 每次使用完全相同的移除集合、初解与修复 seed。",
             "逐步插入后缓存更新受影响的最多四条弧。Regret 比较不同路线结构，方向不同而结构相同的方案仅保留较优者。"
             "有限预算按合法机器人/子块机器人组合分配，位置按 seed 打散；A1 每次局部配置评分计一次预算，故同预算下评估的结构数可能少于 A0。",
             "共同初解复用 REGION 归属/边界逻辑，但通过新可选 cost hook 和关闭旧路线工具绕过内部 DP；从空状态逐块定向追加，锁定合法首方向。"
             "首个 B32 成功初解被所有组共享；失败才尝试其他首方向组合及现有 RULE/X 路线。",
             "新实验无全路线方向 DP、方向 VNS、C2/C3/C4、独立 certifier、精确解或下界调用。每个组内唯一完整候选显式传入四个方向向量，仅调用一次原 B32。"
             "只把 FEASIBLE 的 Cmax 用于比较；基本检查覆盖、资格、固定模式和焊接操作方向。DEADLOCK 不代表数学不可行。",
             "六例为历史已消耗开发几何。沿用 MLP_DEV/load_parents 白名单，未读取新封存测试几何。g15 的历史 ID_TEST 目录不表示当前封存角色。", "",
             "## 共同初解与尝试", ""]
    for name, item in initial_logs.items():
        lines.append(f"- {name}：Cmax={item['cmax']}，耗时 {item['runtime_s']:.3f}s；"
                     f"选区生成 {item['selection_s']:.3f}s；"
                     f"模式 {item['patterns']}；尝试状态 {dict(Counter(a['status'] for a in item['attempts']))}；"
                     f"成功构造 {item['successful_scheme']}。")
        if item["cmax"] is None:
            lines.append("  " + json.dumps(item["attempts"], ensure_ascii=False))
    lines += ["", "## 全部分组结果", "",
              "唯一候选按路线和方向共同去重。最佳值为成功候选中的最小值，可劣于初解；失败留空。仅严格改善候选作为专家保留。"
              "时间包含修复、检查和 B32，选区生成时间另列于初解日志；组间候选可能重合。", "",
              "|实例|方法|q|初始 Cmax|尝试|唯一候选|B32 成功|改善|最佳 Cmax|改善%|秒|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        lines.append(f"|{r['instance']}|{r['method']}|{r['q']}|{fmt(r['initial_cmax'])}|{r['attempts']}|{r['unique_candidates']}|"
                     f"{r['schedule_success']}|{r['improved']}|{fmt(r['best_cmax'])}|{fmt(r['improvement_pct'])}|{r['runtime_s']:.2f}|")
    lines += ["", "## A0/A1 同移除集配对", "",
              "|实例|q|共同成功|A1 胜|平|A0 胜|仅 A1 成功|仅 A0 成功|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in paired:
        lines.append(f"|{p['instance']}|{p['q']}|{p['both_success']}|{p['a1_wins']}|{p['ties']}|{p['a0_wins']}|"
                     f"{p['only_a1_success']}|{p['only_a0_success']}|")
    lines += ["", "## 构造与可调度性", ""]
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        fields = ("attempts", "unique_candidates", "schedule_success", "improved", "restored_original", "changed_assignment",
                  "changed_direction", "reordered", "joint_expert_candidates", "direction_order_only_improved", "movable_blocks", "removed_from_critical_route")
        lines.append(f"- {method}：" + json.dumps({k: sum(r[k] for r in group) for k in fields}, ensure_ascii=False)
                     + "；B32=" + str(dict(sum((Counter(json.loads(r['statuses'])) for r in group), Counter())))
                     + "；修复拒绝=" + str(dict(sum((Counter(json.loads(r['repair_failures'])) for r in group), Counter()))))
    lines += ["", "## 典型成功/失败与可重放动作", ""]
    labels = ('联合轨迹', '仅方向和顺序改善', 'g33_w055:A1_LOAD', 'g33_w055:A1_RANDOM',
              'g36_w086:A1_WAIT_ANCHORED', 'g45_w082:A1_WAIT_ANCHORED')
    selected = {label: cases[label] for label in labels if label in cases}
    selected.update(examples)
    for label, case in selected.items():
        if case is None:
            continue
        c = case["changes"]
        transfers = "; ".join(f"{short(b)} R{a}→R{r}" for b, a, r in c["transfers"]) or "无"
        flips = "; ".join(f"{short(b)} {a}→{d}" for b, a, d in c["flips"]) or "无"
        lines += [f"### {label}：{case['instance']} {case['method']} q{case['q']} attempt={case['attempt']}", "",
                  f"状态 {case['status']}；归属变化：{transfers}；方向变化：{flips}；重排路线 {c['reordered_routes']}。",
                  f"加工负载 {tuple(round(v, 3) for v in c['process_before'])}→{tuple(round(v, 3) for v in c['process_after'])}；"
                  f"空走 {c['travel_before']:.3f}→{c['travel_after']:.3f}s；WAIT {c['wait_before']:.3f}→"
                  + ("无成功调度" if c['wait_after'] is None else f"{c['wait_after']:.3f}s") + "；"
                  f"Cmax {c['cmax_before']}→{c['cmax_after']}。",
                  "移除父焊缝：" + "; ".join(case["removed"]) + "。",
                  "动作（block→R@位置/方向，括号内是邻块 old→new）：" + "; ".join(
                      f"{short(a['block_id'])}→R{a['robot']}@{a['position']}/{a['direction']}"
                      + ("(" + ",".join(f"{short(b)}:{old}→{new}" for b, old, new in a['neighbour_changes']) + ")" if a['neighbour_changes'] else "")
                      for a in case["actions"]) + "。", ""]
        if label == "联合轨迹":
            for r in range(4):
                before = " → ".join(f"{short(b)}/{d}" for b, d in zip(case['before_routes'][r], case['before_directions'][r]))
                after = " → ".join(f"{short(b)}/{d}" for b, d in zip(case['after_routes'][r], case['after_directions'][r]))
                lines.append(f"- R{r}：`{before}` → `{after}`")
    lines += ["", "## X_SPLIT 正确性运行", "", json.dumps(smoke, ensure_ascii=False), "",
              "## 判断", "", *conclusion_lines(rows, paired), "", "## 复现", "", "```powershell",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_joint_oriented_repair_pilot.py "
              f"--instances {' '.join(args.instances)} --attempts {args.attempts} --q {' '.join(map(str,args.q))} "
              f"--seed {args.seed} --insertion-limit {args.insertion_limit} --repair-budget {args.repair_budget}", "```", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--q", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--insertion-limit", type=int, default=8)
    parser.add_argument("--repair-budget", type=int, default=4096)
    args = parser.parse_args()
    if args.attempts < 1 or args.repair_budget < 1:
        parser.error("attempts and repair budget must be positive")
    config = ScientificConfig()
    entries = {e['sheet_name']: e for e in dataset.representatives_v05() if e['split'] == 'MLP_DEV'}
    rows, logs, cases, examples, paired = [], {}, {}, {}, []
    smoke = x_split_smoke(config)
    print("X_SPLIT smoke", smoke['status'], smoke['cmax'], flush=True)
    for name in args.instances:
        parents = dataset.load_parents(entries[name])
        started = time.perf_counter()
        initial, init_attempts = common_initial(parents, config)
        logs[name] = dict(cmax=None if initial is None else initial.schedule.cmax, runtime_s=time.perf_counter()-started,
                          attempts=init_attempts, patterns={} if initial is None else dict(Counter(p.kind.value for p in initial.solution.patterns)),
                          successful_scheme=next((a['scheme'] for a in init_attempts if a['status']=='FEASIBLE'), None), selection_s=0)
        print("INITIAL", name, "Cmax", logs[name]['cmax'], "scheme", logs[name]['successful_scheme'],
              "attempts", dict(Counter(a['status'] for a in init_attempts)),
              "time", round(logs[name]['runtime_s'], 3), flush=True)
        for q in args.q:
            selection_start = time.perf_counter()
            partials = {}
            for selection in BottleneckSelection:
                selected = []
                if initial is not None:
                    for attempt in range(args.attempts):
                        seed = args.seed + NAMES.index(name)*10000 + q*100 + attempt
                        p = select_bottleneck_parents(initial.solution, initial.directions, initial.schedule, selection,
                                                       q=q, seed=seed, config=config)
                        if p is None:
                            break
                        selected.append(p)
                partials[selection.value] = selected
            logs[name]['selection_s'] += time.perf_counter()-selection_start
            outcomes = {}
            for method in METHODS:
                selection = method.split('_', 1)[1]
                row, best, outcome, example = run_group(name, method, q, partials[selection], initial, args, config)
                rows.append(row)
                outcomes[method] = outcome
                if best is not None:
                    label = f"{name}:{method}"
                    if label not in cases or best['changes']['cmax_after'] < cases[label]['changes']['cmax_after']:
                        cases[label] = best
                    c = best['changes']
                    if c['transfers'] and c['flips'] and c['reordered_routes']:
                        cases.setdefault('联合轨迹', best)
                    if not c['transfers'] and c['flips'] and c['reordered_routes']:
                        cases.setdefault('仅方向和顺序改善', best)
                if example is not None and name in ('g33_w055','g36_w086','g45_w082'):
                    examples.setdefault(f"{name}:失败或未改善", example)
            a0, a1 = outcomes['A0_RANDOM'], outcomes['A1_RANDOM']
            both = [(a0[i][1], a1[i][1]) for i in a0.keys() & a1.keys() if a0[i][1] is not None and a1[i][1] is not None]
            paired.append(dict(instance=name, q=q, shared_attempts=len(a0.keys() & a1.keys()), both_success=len(both),
                               a1_wins=sum(b<a-1e-6 for a,b in both), ties=sum(abs(a-b)<=1e-6 for a,b in both),
                               a0_wins=sum(a<b-1e-6 for a,b in both),
                               only_a1_success=sum(a1[i][1] is not None and a0[i][1] is None for i in a0.keys() & a1.keys()),
                               only_a0_success=sum(a0[i][1] is not None and a1[i][1] is None for i in a0.keys() & a1.keys())))
        # Keep representative cases, not every non-improving full candidate.
        save(rows, logs, cases, examples, paired, smoke, args)
    print(f"Saved {CSV_PATH}\nSaved {REPORT_PATH}", flush=True)


if __name__ == '__main__':
    main()
