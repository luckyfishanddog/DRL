from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.model import ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi


def _checkpoint(result, deadline: float, budget: float) -> dict[str, object]:
    if deadline > budget:
        return {"cmax": None, "reason": "CHECKPOINT_BEYOND_REQUESTED_BUDGET"}
    return result.anytime[deadline]


def _run(
    entry: dict[str, Any], root: Path, budget: float, source_commit_label: str
) -> dict[str, Any]:
    instance = load_ppo_platform_instance(
        root / entry["relative_path"],
        entry["sheet_name"],
        ppo_root=root,
        instance_id=entry["instance_id"],
    )
    parents = to_parent_welds(instance)
    result = run_bounded_sa_oi(
        parents,
        ScientificConfig(delta_x=0.20, delta_y=0.20),
        SearchConfig(time_limit=budget),
        seed=20260929,
        scope=FORMAL_SCOPE_V1_1,
        source_commit=source_commit_label,
        allow_unverified_source=True,
        formal_result=False,
    )
    records = result.stats.reference_records
    initial_schedule = result.initialization.schedule
    final_certified = bool(
        result.final_certification and result.final_certification.certified
    )
    return {
        "tier": entry["tier"],
        "instance_id": entry["instance_id"],
        "actual_weld_count": entry["actual_weld_count"],
        "total_weld_length_m": entry["total_weld_length_m"],
        "dataset_manifest_hash": entry["dataset_manifest_hash"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "solver_seed": 20260929,
        "requested_time_limit_s": budget,
        "initialization_status": result.initialization.status.value,
        "initialization_strategy": result.initialization.winning_strategy,
        "initialization_time_s": result.stats.init_time,
        "construction_attempts": result.stats.construction_attempts,
        "initial_cmax": None if initial_schedule is None else initial_schedule.cmax,
        "best_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "cmax_at_1": _checkpoint(result, 1.0, budget),
        "cmax_at_5": _checkpoint(result, 5.0, budget),
        "cmax_at_30": _checkpoint(result, 30.0, budget),
        "iterations": result.stats.iterations,
        "Nref": result.stats.nref,
        "init_reference_calls": result.stats.init_reference_calls,
        "total_reference_calls": len(records),
        "baseline_DEADLOCK": sum(bool(item["baseline_deadlock"]) for item in records),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in records
        ),
        "remaining_DEADLOCK": sum(item["status"] == "DEADLOCK" for item in records),
        "actual_runtime_s": result.runtime,
        "overshoot_s": result.stats.overshoot,
        "scheduler_time_s": result.stats.reference_scheduler_time,
        "repair_time_s": result.stats.repair_time,
        "certifier_time_s": result.stats.certifier_time,
        "final_schedule_status": (
            None if result.best_schedule is None else result.best_schedule.status.value
        ),
        "final_certification": final_certified,
        "certification_errors": (
            []
            if result.final_certification is None
            else list(result.final_certification.errors)
        ),
        "search_status": result.status.value,
    }



# Current finite-range development sensitivity. Historical smoke above keeps 0.2.
_RANGE_GROUPS = {"A": (0.2, 0.2), "D": (0.5, 0.5)}  # delta_y, delta_x
_RANGE_SEEDS = (20261081, 20261082)
_RANGE_MARKER = "<!-- YX_RANGE_RESULTS -->"


def _range_entries(root):
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    selected = json.loads((root / "data/manifests/PPO_X_SPLIT_GATE_SET_V1.json").read_text(encoding="utf-8"))
    entries = [e for e in selected["instances"] if e["selection_ordinal"] in (2, 3, 5, 6, 9, 12)]
    entries.sort(key=lambda e: e["selection_ordinal"])
    roles = json.loads((root / "data/manifests/PPO_V2_DATA_ROLES_V1.json").read_text(encoding="utf-8"))
    assert_v2_solver_access_allowed(roles, [e["relative_path"] for e in entries],
                                    allowed_roles=("V2_MODEL_DEVELOPMENT_CONSUMED",))
    return entries


def _range_parents(entry, ppo):
    return to_parent_welds(load_ppo_platform_instance(
        ppo / entry["relative_path"], entry["sheet_name"], ppo_root=ppo,
        instance_id=entry["instance_id"]))


def _range_config(group):
    y, x = _RANGE_GROUPS[group]
    return ScientificConfig(delta_y=y, delta_x=x)


def _range_catalog(entries, ppo):
    from dataclasses import replace
    from mrta_reference.geometry import (build_legal_pattern_catalog, frozen_handover_centers,
        whole_eligible_rails, generate_y_split_patterns, generate_x_split_patterns)
    from mrta_reference.model import Rail, SplitKind
    from mrta_reference.scope import FORMAL_SCOPE_V2
    rows = []
    for entry in entries:
        parents = _range_parents(entry, ppo)
        for group in _RANGE_GROUPS:
            cfg = _range_config(group)
            catalog = build_legal_pattern_catalog(parents, cfg, FORMAL_SCOPE_V2)
            upper, lower = frozen_handover_centers(parents, cfg)
            loose = replace(cfg, min_child_length=1e-12)
            row = dict(instance=entry["selection_ordinal"], tier=entry["tier"], group=group,
                       N=len(parents), whole=0, mandatory_y=0, optional_y=0,
                       y_patterns=0, x_parents=0, x_patterns=0, no_split=0,
                       removed_short_y=0, removed_short_x=0, x_up=upper, x_low=lower)
            for parent in parents:
                patterns = catalog[parent.parent_id]
                eligible = whole_eligible_rails(parent.start, parent.end, cfg)
                ys = sum(p.kind is SplitKind.Y_SPLIT for p in patterns)
                xs = sum(p.kind is SplitKind.X_SPLIT for p in patterns)
                row["whole"] += bool(eligible)
                row["mandatory_y"] += not bool(eligible)
                row["optional_y"] += bool(eligible) and ys > 0
                row["y_patterns"] += ys
                row["x_parents"] += xs > 0
                row["x_patterns"] += xs
                row["no_split"] += not (ys or xs)
                row["removed_short_y"] += len(generate_y_split_patterns(parent, loose)) - ys
                for rail, center in ((Rail.UPPER, upper), (Rail.LOWER, lower)):
                    row["removed_short_x"] += (len(generate_x_split_patterns(parent, center, loose, rail=rail))
                        - len(generate_x_split_patterns(parent, center, cfg, rail=rail)))
            rows.append(row)
    return rows


def _range_worker_init(queue):
    from mrta_ranker.dataset import bind_to_core
    global _RANGE_CORE_MASK
    _RANGE_CORE_MASK = queue.get()
    bind_to_core(_RANGE_CORE_MASK)


def _range_worker(job):
    from collections import Counter
    from mrta_ranker.dataset import bind_to_core
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry, seed, group, ppo, mask = job
    mask = _RANGE_CORE_MASK
    cfg = _range_config(group)
    parents = _range_parents(entry, Path(ppo))
    snapshots = []
    last_solution = None
    def observe(event, value):
        nonlocal last_solution
        if event == "boundary" and value["solution"] is not last_solution:
            last_solution = value["solution"]
            snapshots.append((value["solution"], value["schedule"], value["metrics"]))
    result = run_sa_oi_alns_v2(parents, cfg, SearchConfig(
        m=192, m_lns=48, kdp=8, kref=2, kref_total=4,
        construction_budget=5, kinit_ref=5, feasibility_bootstrap_budget=1,
        max_iterations=100000, time_limit=60., checkpoints=(5., 30., 60.),
        enable_two_opt_star=False), seed=seed, source_commit="LOCAL_YX_RANGE_DEVELOPMENT",
        allow_unverified_source=True, formal_result=False, observer=observe, ranker=None)
    c60 = result.anytime[60.]["cmax"]
    at60 = next(((s, t, m) for s, t, m in snapshots if c60 is not None and
                 abs(m.cmax - c60) <= 1e-8), None)
    certification = None
    checkpoint = {}
    if at60 is not None:
        solution, schedule, metrics = at60
        certification = certify_schedule(solution, schedule, cfg, scope=FORMAL_SCOPE_V2)
        checkpoint = dict(patterns=dict(Counter(p.kind.value for p in solution.patterns)),
            process_imbalance=metrics.process_imbalance,
            empty_travel_s=metrics.total_empty_travel, waiting_s=metrics.total_waiting)
    stats = result.stats
    return dict(instance=entry["selection_ordinal"], instance_id=entry["instance_id"],
        tier=entry["tier"], seed=seed, group=group, core_mask=mask,
        certified_at60=bool(certification and certification.certified), cmax_at60=c60,
        checkpoint_errors=[] if certification is None else list(certification.errors),
        **checkpoint, iterations=stats.iterations, reference_calls=stats.nref+stats.init_reference_calls,
        raw_attempts=stats.raw_attempts, valid_candidates=sum(stats.valid_by_family.values()),
        candidate_valid_rate=sum(stats.valid_by_family.values())/max(1, stats.raw_attempts),
        deadlock=stats.n_deadlock, infeasible=stats.n_infeasible, numeric_failure=stats.n_numeric_failure,
        init_status_counts=dict(stats.init_status_counts), accepted_moves=sum(stats.accepted_by_family.values()),
        global_best_updates=max(0,len(stats.best_events)-1), elapsed_s=result.runtime,
        overshoot_s=stats.overshoot, status=result.status.value,
        final_certified=bool(result.final_certification and result.final_certification.certified),
        final_errors=[] if result.final_certification is None else list(result.final_certification.errors),
        final_cmax=None if result.best_metrics is None else result.best_metrics.cmax,
        final_patterns={} if result.best_solution is None else dict(Counter(p.kind.value for p in result.best_solution.patterns)),
        max_kdp=max(stats.per_iteration_kdp,default=0), max_reference_per_iteration=max(stats.per_iteration_nref,default=0))


def _range_table(headers, rows):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


def _range_write(report, payload):
    from statistics import mean, median
    results = payload["runs"]
    text = "# Y/X 拆分范围 0.5 m：正确性与 development sensitivity\n\n"
    text += ("当前默认 delta_y=delta_x=0.50 m；最短子焊缝仍为 0.20 m；干涉距离仍为 0.50 m。"
        "Y 候选为 5.5/5.8/6.0/6.2/6.5，X 为各轨固定中心 ±0.5/±0.2/0，保留合法 MIDPOINT。"
        "旧点身份保留，新增点使用 OUTER_LOWER/OUTER_UPPER；几何重复去重。按用户最新指示，本轮只保留 A/D 对比，不分析 B/C 单变量方案。\n\n"
        "覆盖区域是任务层假设：上轨覆盖 y≥5.5，下轨覆盖 y≤6.5；未做 URDF/IK 物理可达性验证。"
        "Y 覆盖改变 WHOLE/mandatory Y 及 X 中心，A/D 不是同一可行域下的纯算法竞赛。\n\n"
        "Phase3-Z、Phase4-0/0B、Phase4-1A/B/C 均保持历史 0.2 配置身份；旧标签、模型与结论未重写。"
        "本轮不训练 MLP/GAT、不采集标签，17 TRAIN / 6 DEV 划分未改变。\n\n"
        "## 实验设置\n\n"
        "A=(delta_y 0.2, delta_x 0.2)，D=(delta_y 0.5, delta_x 0.5)。"
        "六例为既有 Phase4-1A 的 I2/I3、I5/I6、I9/I12，每档两例，均为 V2_MODEL_DEVELOPMENT_CONSUMED；"
        "读取前检查既有数据角色，不访问 validation、ORACLE 或 sealed ID_TEST。"
        "seeds=20261081、20261082，实例与顺序在求解前固定。\n\n"
        "SA_OI_ALNS_V2；M192=144 atomic+48 LNS；Kdp8/Kref2/Kref_total4；heuristic C2/C4；"
        "TWO_OPT_STAR OFF；B32、SA、direction 与候选生成器其余设置一致。"
        "初始化复用此前 construction_budget=5 / kinit_ref=5 / bootstrap=1。"
        "每 run 墙钟 60 s，四个 worker 各绑定一个物理核的一个逻辑线程，A/D 轮换顺序。"
        "特征/选择/观察记录耗时计入预算。Cmax@60 使用原生 anytime，绝不用 overshoot final_cmax 决胜。"
        "@60 解另做独立 certifier 复核；计数/负载/行走/等待均对应该解，final_patterns 另存原始记录。\n\n")
    text += "完整测试：" + payload.get("tests", "待完成") + "。\n\n"
    text += "| 实例 | 档位 | workbook / sheet |\n| --- | --- | --- |\n"
    for e in payload["entries"]:
        text += f"| I{e['selection_ordinal']} | {e['tier']} | {e['relative_path']} / {e['sheet_name']} |\n"
    text += "\n## 静态目录（零 reference 调用）\n\n"
    headers = ["实例", "组", "WHOLE", "mandatory Y", "optional Y", "Y patterns", "X parents", "X patterns", "无合法拆分", "短 Y 删除", "短 X 删除", "x_up", "x_low"]
    keys = ["instance","group","whole","mandatory_y","optional_y","y_patterns","x_parents","x_patterns","no_split","removed_short_y","removed_short_x","x_up","x_low"]
    text += _range_table(headers, [[r[k] for k in keys] for r in payload["catalog"]])
    text += ("\n\nWHOLE/mandatory/optional 统计父焊缝；X patterns 按轨计数。无合法拆分表示仅 WHOLE 可选，"
             "不等于无可行任务。短段删除为同一候选生成器在去除 0.2 m 下限前后、按几何去重后的差值。"
             "纵向/端点/区域外候选不计为短段删除。固定中心的旧 ±0.2 偏移保留；Y 资格改变时中心可移动，"
             "不能声称 A 的所有绝对 X 切点都嵌套于 D。\n\n## 60 秒实验\n\n")
    text += f"已完成 {len(results)}/24。\n\n"
    if results:
        summaries=[]
        for tier in ("SMALL","MEDIUM","LARGE"):
            for group in _RANGE_GROUPS:
                rs=[r for r in results if r["tier"]==tier and r["group"]==group]
                ok=[r for r in rs if r.get("certified_at60")]
                if not rs: continue
                avg=lambda k: mean(r[k] for r in ok) if ok else None
                summaries.append([tier,group,len(ok),len(rs)-len(ok),avg("cmax_at60"),
                    avg("iterations"),avg("reference_calls"),avg("candidate_valid_rate"),
                    avg("process_imbalance"),avg("empty_travel_s"),avg("waiting_s"),avg("overshoot_s")])
        text+=_range_table(["档位","组","认证","未认证","Cmax均值","迭代均值","ref均值","有效率","负载差","空行走s","等待s","overshoot s"],summaries)
        text+="\n\n配对差以同实例同 seed 的 A 为参照，负数表示 Cmax 改善。均值只包含双方 @60 认证的完整配对。\n\n"
        pairs=[]
        indexed={(r['instance'],r['seed'],r['group']):r for r in results}
        for tier in ("SMALL","MEDIUM","LARGE","ALL"):
            for group in ("D",):
                ps=[]
                for r in results:
                    if r["group"]!=group or (tier!="ALL" and r["tier"]!=tier): continue
                    a=indexed.get((r['instance'],r['seed'],'A'))
                    if a and a.get('certified_at60') and r.get('certified_at60'):
                        ps.append((r['cmax_at60']-a['cmax_at60'],100*(r['cmax_at60']/a['cmax_at60']-1)))
                if ps: pairs.append([tier,group,len(ps),mean(p[0] for p in ps),median(p[1] for p in ps),sum(p[0]<-1e-8 for p in ps),sum(abs(p[0])<=1e-8 for p in ps),sum(p[0]>1e-8 for p in ps)])
        text+=_range_table(["档位","组-A","配对数","Cmax差均值s","相对差中位%","改善","相同","变差"],pairs)
        text+="\n\n逐次结果（WHOLE/Y/X 为 @60 解；空行走与等待单位秒）：\n\n"
        text+=_range_table(["I","seed","组","认证","Cmax@60","迭代","ref","有效率","WHOLE/Y/X","负载差","空行走","等待","elapsed","overshoot"],[[r['instance'],r['seed'],r['group'],r.get('certified_at60'),r.get('cmax_at60'),r.get('iterations'),r.get('reference_calls'),r.get('candidate_valid_rate'),'/'.join(str(r.get('patterns',{}).get(k,0)) for k in ('WHOLE','Y_SPLIT','X_SPLIT')),r.get('process_imbalance'),r.get('empty_travel_s'),r.get('waiting_s'),r.get('elapsed_s'),r.get('overshoot_s')] for r in sorted(results,key=lambda r:(r['instance'],r['seed'],r['group']))])
    text += "\n\n" + payload.get("conclusions", "最终分析将在 24 次实际运行完成后补充。")
    text += "\n\n## 可恢复的逐次原始记录\n\n字段中 final_cmax/final_patterns 可能来自 overshoot，仅用于审计，不用于上述质量比较。\n\n"
    text += _RANGE_MARKER + "\n```json\n" + json.dumps(payload,ensure_ascii=False,indent=2) + "\n```\n"
    if report.exists():
        previous=report.read_text(encoding="utf-8")
        if "\n\n## SMALL_30_39复核" in previous:
            text += "\n\n## SMALL_30_39复核" + previous.split("\n\n## SMALL_30_39复核",1)[1]
    report.write_text(text,encoding="utf-8")


def _range_main(action, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from mrta_ranker.dataset import physical_core_affinities
    root=Path(__file__).resolve().parents[1]
    ppo=root.parent/"ppo"
    report=root/"docs/YX_SPLIT_RANGE_05_CHANGE.md"
    if report.exists():
        content=report.read_text(encoding="utf-8")
        payload=json.loads(content.split(_RANGE_MARKER+"\n```json\n",1)[1].split("\n```",1)[0])
    else:
        entries=_range_entries(root)
        payload=dict(entries=entries,catalog=_range_catalog(entries,ppo),runs=[],tests="待完成")
    _range_write(report,payload)
    if action=="catalog":
        print(_range_table(["组","WHOLE","mandatoryY","optionalY","Ypatterns","Xparents","Xpatterns","shortY","shortX"],[[g]+[sum(r[k] for r in payload['catalog'] if r['group']==g) for k in ('whole','mandatory_y','optional_y','y_patterns','x_parents','x_patterns','removed_short_y','removed_short_x')] for g in _RANGE_GROUPS]),flush=True)
        return 0
    if action=="report": return 0
    entries=_range_entries(root)  # Preserve existing role restriction on resumed runs too.
    masks=physical_core_affinities()[:workers]
    done={(r['instance'],r['seed'],r['group']) for r in payload['runs']}
    jobs=[]
    for i,e in enumerate(entries):
        for j,seed in enumerate(_RANGE_SEEDS):
            groups=list(_RANGE_GROUPS)
            offset=(i+j)%len(groups)
            for group in groups[offset:]+groups[:offset]:
                if (e['selection_ordinal'],seed,group) not in done:
                    jobs.append((e,seed,group,str(ppo),masks[len(jobs)%len(masks)]))
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    core_queue = context.Queue()
    for mask in masks:
        core_queue.put(mask)
    with ProcessPoolExecutor(max_workers=len(masks), mp_context=context,
                             initializer=_range_worker_init, initargs=(core_queue,)) as pool:
        pending={pool.submit(_range_worker,job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try: row=future.result()
            except Exception as exc:
                row=dict(instance=job[0]['selection_ordinal'],tier=job[0]['tier'],seed=job[1],group=job[2],certified_at60=False,error=repr(exc))
            payload['runs'].append(row)
            _range_write(report,payload)
            print(f"Completed {len(payload['runs'])}/24: I{row['instance']} {row['seed']} {row['group']} certified={row.get('certified_at60')} Cmax@60={row.get('cmax_at60')}",flush=True)
    return 0 if len(payload['runs'])==24 else 2



_SMALL30_MARKER = "<!-- SMALL30_RESULTS -->"
_SMALL30_SEEDS = (20261081, 20261082, 20261083)


def _small30_choose(candidates):
    """Static geometry only: nearest N=35, rich / sparse / median, distinct books."""
    distance = min(abs(e["N"] - 35) for e in candidates)
    pool = [e for e in candidates if abs(e["N"] - 35) == distance]
    def stable(e): return e["relative_path"], e["sheet_name"]
    rich = min(pool, key=lambda e: (-e["x_patterns"], -e["y_patterns"], stable(e)))
    rest = [e for e in pool if e["relative_path"] != rich["relative_path"]]
    sparse = min(rest, key=lambda e: (e["x_patterns"]+e["y_patterns"], e["x_patterns"], stable(e)))
    rest = [e for e in rest if e["relative_path"] != sparse["relative_path"]]
    from statistics import median
    mid = median(e["x_patterns"] for e in pool)
    ordinary = min(rest, key=lambda e: (abs(e["x_patterns"]-mid), stable(e)))
    return [dict(e, selection_ordinal=i+1, tier="SMALL_30_39", geometry_role=role)
            for i, (e, role) in enumerate(((rich,"X_RICH"),(ordinary,"ORDINARY"),(sparse,"SPLIT_SPARSE")))]


def _small30_select(root, ppo):
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from mrta_data.ppo_instances import inspect_ppo_workbook
    from mrta_reference.geometry import (build_legal_pattern_catalog, whole_eligible_rails,
        frozen_handover_centers)
    from mrta_reference.model import Rail, SplitKind
    from mrta_reference.scope import FORMAL_SCOPE_V2
    roles=json.loads((root/"data/manifests/PPO_V2_DATA_ROLES_V1.json").read_text(encoding="utf-8"))
    paths=sorted(e["relative_path"] for e in roles["workbooks"]
                 if e["new_v2_role"]=="V2_MODEL_DEVELOPMENT_CONSUMED")
    assert_v2_solver_access_allowed(roles,paths,allowed_roles=("V2_MODEL_DEVELOPMENT_CONSUMED",))
    manifest=json.loads((root/"data/manifests/PPO_DATASET_MANIFEST_V1.json").read_text(encoding="utf-8"))
    valid={(e["relative_path"],e["sheet_name"]):e for e in manifest["instances"]
           if e["relative_path"] in paths and e["validation_status"]=="VALID" and not e.get("duplicate_of")}
    eligible=[]
    for path in paths:
        inspection=inspect_ppo_workbook(ppo/path,ppo_root=ppo)
        if inspection.errors: raise ValueError((path,inspection.errors))
        for sheet in inspection.sheets:
            if sheet.name=="META": continue
            meta=inspection.sheet_metadata[sheet.name]
            count=int(meta.get("actual_weld_count") or meta["actual_N"])
            if count!=sheet.row_count: raise ValueError("sheet META / actual rows disagree")
            entry=valid.get((path,sheet.name))
            if entry is not None and count!=entry["actual_weld_count"]:
                raise ValueError("existing intake count disagrees with current workbook")
            if entry is not None and 30<=count<=39:
                eligible.append(entry)
    nearest=min(abs(e["actual_weld_count"]-35) for e in eligible)
    pool=[e for e in eligible if abs(e["actual_weld_count"]-35)==nearest]
    candidates=[]
    cfg=_range_config("D")
    for entry in pool:
        parents=_range_parents(entry,ppo)
        if len(parents)!=entry["actual_weld_count"]: raise ValueError("actual N changed")
        catalog=build_legal_pattern_catalog(parents,cfg,FORMAL_SCOPE_V2)
        rail_counts={"upper_only":0,"lower_only":0,"both":0,"mandatory_y":0}
        for parent in parents:
            rails=whole_eligible_rails(parent.start,parent.end,cfg)
            key="mandatory_y" if not rails else "both" if len(rails)==2 else "upper_only" if Rail.UPPER in rails else "lower_only"
            rail_counts[key]+=1
        upper,lower=frozen_handover_centers(parents,cfg)
        candidates.append(dict(instance_id=entry["instance_id"],relative_path=entry["relative_path"],
            sheet_name=entry["sheet_name"],N=len(parents),
            total_length_m=sum(p.length for p in parents),
            x_patterns=sum(p.kind is SplitKind.X_SPLIT for ps in catalog.values() for p in ps),
            y_patterns=sum(p.kind is SplitKind.Y_SPLIT for ps in catalog.values() for p in ps),
            x_parents=sum(any(p.kind is SplitKind.X_SPLIT for p in ps) for ps in catalog.values()),
            cross_y6=sum(min(p.start[1],p.end[1])<6<max(p.start[1],p.end[1]) for p in parents),
            cross_full_by=sum(min(p.start[1],p.end[1])<5.5 and max(p.start[1],p.end[1])>6.5 for p in parents),
            x_up=upper,x_low=lower,**rail_counts,
            bbox_area_ratio=entry["bbox_area_ratio"],grid_occupancy_ratio=entry["grid_occupancy_ratio"],
            upper_lower_length_imbalance=entry["upper_lower_length_imbalance"],
            quadrant_length_cv=entry["quadrant_length_cv"]))
    entries=_small30_choose(candidates)
    return dict(entries=entries,candidates=candidates,eligible_sheet_count=len(eligible),
        eligible_workbook_count=len({e['relative_path'] for e in eligible}),
        inspected_development_workbooks=len(paths),runs=[],i3_diagnostics=[],
        selection_rule="META actual_weld_count equals actual rows; VALID unique consumed-development only; closest N to35; maximum legal X patterns, minimum X+Y patterns, median X patterns; three distinct workbooks; ties path/sheet; no solver/Cmax fields")



def _range_search_config(*, budget=60., iterations=100000):
    return SearchConfig(m=192,m_atomic=144,m_lns=48,kdp=8,kref=2,kref_total=4,
        construction_budget=5,kinit_ref=5,feasibility_bootstrap_budget=1,
        max_iterations=iterations,time_limit=budget,checkpoints=(5.,30.,60.),
        enable_two_opt_star=False)


def _small30_solution_details(solution,schedule,metrics,cfg):
    from collections import Counter
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    blocks=block_map(solution,cfg,scope=FORMAL_SCOPE_V2)
    loads=[sum(cfg.process_time(blocks[b].length) for b in r.block_ids) for r in solution.routes]
    parents={p.parent_id:p for p in solution.parents}
    outer=[dict(parent=p.parent_id,kind=p.kind.value,point_id=p.point_id,
                rail=None if p.rail is None else p.rail.value,t=p.t,
                coordinate=parents[p.parent_id].point(p.t))
           for p in solution.patterns if p.point_id and "_OUTER_" in p.point_id]
    return dict(cmax=metrics.cmax,patterns=dict(Counter(p.kind.value for p in solution.patterns)),
        process_loads=loads,process_imbalance=metrics.process_imbalance,
        empty_travel_s=metrics.total_empty_travel,waiting_s=metrics.total_waiting,outer_cuts=outer,
        solution=dict(pattern_ids=[p.pattern_id for p in solution.patterns],
                      routes=[list(r.block_ids) for r in solution.routes]))


def _small30_pick_snapshot(snapshots,cmax):
    eligible=[row for row in snapshots if row[0]<=60. and cmax is not None
              and abs(row[3].cmax-cmax)<=1e-8]
    return eligible[-1] if eligible else None


class _Small30Telemetry:
    """Worker-local observation of existing counters; no reference evaluation added."""
    def __init__(self):
        self.stats=None;self.references=[];self.first_init=None;self.initial=None
        self.best=[];self.best_count=0;self.at60={};self.current_cmax=None
        self.c4_feasible=0;self.c4_improving=0;self.c4_material=0
        self.proposal_improving=0

    def __enter__(self):
        import time
        from mrta_search.stats import SearchStats
        self.original=SearchStats.record_reference
        def record(stats,status,duration,**kwargs):
            self.original(stats,status,duration,**kwargs)
            self.stats=stats
            schedule=kwargs.get("schedule")
            row=dict(elapsed=time.perf_counter()-stats.run_started,
                initialization=kwargs["initialization"],status=status.value,
                cmax=None if schedule is None else schedule.cmax,
                diagnostics=[] if schedule is None else list(schedule.diagnostics))
            self.references.append(row)
            # Initialization records FEASIBLE only after independent certification.
            if row["initialization"] and row["status"]=="FEASIBLE" and self.first_init is None:
                self.first_init=row.copy()
        SearchStats.record_reference=record
        return self

    def __exit__(self,*args):
        from mrta_search.stats import SearchStats
        SearchStats.record_reference=self.original

    def __call__(self,event,value):
        if event=="iteration_evaluation":
            _,c4,_=value
            for evaluated in c4:
                if evaluated.metrics is not None:
                    self.c4_feasible+=1
                    gain=(self.current_cmax-evaluated.metrics.cmax)/self.current_cmax
                    self.c4_improving+=gain>1e-9
                    self.c4_material+=gain>=.005
        elif event=="post_c4":
            self.proposal_improving+=value.metrics.cmax<self.current_cmax-1e-9
        elif event=="boundary":
            stats=self.stats
            if self.initial is None:
                self.initial=(value['solution'],value['schedule'],value['metrics'])
            if len(stats.best_events)>self.best_count:
                self.best.append((stats.best_events[-1][0],value['solution'],value['schedule'],value['metrics']))
                self.best_count=len(stats.best_events)
            self.current_cmax=value['metrics'].cmax
            if value['elapsed']<=60.:
                self.at60=dict(iterations=value['iteration'],accepted_moves=sum(stats.accepted_by_family.values()),
                    raw_attempts=stats.raw_attempts,valid_candidates=sum(stats.valid_by_family.values()),
                    global_best_updates=max(0,len(stats.best_events)-1),
                    c4_feasible=self.c4_feasible,c4_improving=self.c4_improving,
                    c4_material=self.c4_material,proposal_improving=self.proposal_improving,
                    decision_family_funnel={k:dict(v) for k,v in stats.decision_family_funnel.items()},
                    timing_s={k:getattr(stats,k) for k in ('candidate_generation_time','cheap_screen_time',
                        'direction_dp_time','reference_scheduler_time','certifier_time','repair_time')})


def _small30_worker(job):
    from collections import Counter
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry,seed,group,ppo,_=job
    cfg=_range_config(group);parents=_range_parents(entry,Path(ppo))
    with _Small30Telemetry() as trace:
        result=run_sa_oi_alns_v2(parents,cfg,_range_search_config(),seed=seed,
            source_commit="LOCAL_SMALL30_DEVELOPMENT",allow_unverified_source=True,
            formal_result=False,observer=trace,ranker=None)
    cmax=result.anytime[60.]['cmax']
    # Best-event acceptance may precede a boundary by a few microseconds.
    trace.at60['global_best_updates']=max(0,sum(t<=60. for t,_ in result.stats.best_events)-1)
    chosen=_small30_pick_snapshot(trace.best,cmax)
    certification=None;at60=None
    if chosen is not None:
        _,solution,schedule,metrics=chosen
        certification=certify_schedule(solution,schedule,cfg,scope=FORMAL_SCOPE_V2)
        at60=_small30_solution_details(solution,schedule,metrics,cfg)
    initial=None
    if trace.initial is not None:
        initial=_small30_solution_details(*trace.initial,cfg)
    before=[r for r in trace.references if r['elapsed']<=60.]
    counts=Counter(r['status'] for r in before)
    rejected=Counter((r['status'],str(r['diagnostics'])) for r in before if r['status']!='FEASIBLE')
    stats=result.stats
    return dict(instance=entry['selection_ordinal'],instance_id=entry['instance_id'],
        workbook=entry['relative_path'],sheet=entry['sheet_name'],N=len(parents),tier='SMALL_30_39',
        seed=seed,group=group,core_mask=_RANGE_CORE_MASK,first_certified_initial=trace.first_init,
        chosen_initial=initial,init_time_s=stats.init_time,init_strategy=result.initialization.winning_strategy,
        certified_at60=bool(certification and certification.certified),cmax_at60=cmax,at60=at60,
        checkpoint_errors=[] if certification is None else list(certification.errors),
        counters_at60=trace.at60,reference_calls_at60=len(before),reference_status_counts_at60=dict(counts),
        init_status_counts=dict(stats.init_status_counts),
        rejection_diagnostics_at60=[dict(status=k[0],diagnostic=k[1],count=v) for k,v in rejected.items()],
        candidate_valid_rate_at60=trace.at60.get('valid_candidates',0)/max(1,trace.at60.get('raw_attempts',0)),
        best_events=[list(e) for e in stats.best_events],
        iterations_total=stats.iterations,accepted_moves_total=sum(stats.accepted_by_family.values()),
        global_best_updates_total=max(0,len(stats.best_events)-1),reference_calls_total=stats.nref+stats.init_reference_calls,
        reference_status_counts_total=dict(Counter(r['status'] for r in trace.references)),
        elapsed_s=result.runtime,overshoot_s=stats.overshoot,status=result.status.value,
        final_cmax=None if result.best_metrics is None else result.best_metrics.cmax,
        final_certified=bool(result.final_certification and result.final_certification.certified),
        final_errors=[] if result.final_certification is None else list(result.final_certification.errors),
        max_kdp=max(stats.per_iteration_kdp,default=0),max_reference_per_iteration=max(stats.per_iteration_nref,default=0))


def _small30_i3(root,ppo):
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.solution import canonicalize,official_metrics
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry=next(e for e in _range_entries(root) if e['selection_ordinal']==3)
    parents=_range_parents(entry,ppo);results={};rows=[]
    for group in ('A','D'):
        cfg=_range_config(group)
        with _Small30Telemetry() as trace:
            result=run_sa_oi_alns_v2(parents,cfg,_range_search_config(budget=None,iterations=0),
                seed=20261081,source_commit="LOCAL_I3_INITIALIZATION_DIAGNOSTIC",
                allow_unverified_source=True,formal_result=False,observer=trace)
        results[group]=result
        details=None if result.best_solution is None else _small30_solution_details(
            result.best_solution,result.best_schedule,result.best_metrics,cfg)
        rows.append(dict(kind='INITIALIZATION_ONLY',group=group,seed=20261081,N=26,
            first_certified_initial=trace.first_init,chosen_initial=details,init_time_s=result.stats.init_time,
            strategy=result.initialization.winning_strategy,reference_calls=result.stats.init_reference_calls,
            certified=bool(result.final_certification and result.final_certification.certified)))
    old=results['A'];cfg=_range_config('D');catalog=build_legal_pattern_catalog(parents,cfg,FORMAL_SCOPE_V2)
    mapped=[];missing=[]
    if old.best_solution is None: return rows
    for pattern in old.best_solution.patterns:
        match=next((p for p in catalog[pattern.parent_id] if p.kind==pattern.kind and p.rail==pattern.rail
                    and ((p.t is None and pattern.t is None) or (p.t is not None and pattern.t is not None
                         and abs(p.t-pattern.t)<=cfg.numeric_epsilon))),None)
        if match is None: missing.append(pattern.pattern_id)
        else: mapped.append(match)
    check=dict(kind='A_INITIAL_REBUILT_IN_D',missing_patterns=missing,certified=False)
    if not missing:
        try:
            solution=canonicalize(parents,mapped,old.best_solution.routes,cfg,scope=FORMAL_SCOPE_V2)
            # A catalog check alone is insufficient: perform one normal reference evaluation.
            schedule=FormalReferenceEvaluator(FORMAL_SCOPE_V2)(solution,cfg,
                orientations={r:old.best_directions[r] for r in range(4)})
            certification=certify_schedule(solution,schedule,cfg,scope=FORMAL_SCOPE_V2)
            check.update(status=schedule.status.value,certified=certification.certified,
                certification_errors=list(certification.errors),reference_calls=1)
            if certification.certified:
                check['details']=_small30_solution_details(solution,schedule,official_metrics(solution,schedule,cfg),cfg)
        except ValueError as exc: check['error']=str(exc)
    rows.append(check)
    return rows



def _small30_read(report):
    content=report.read_text(encoding="utf-8")
    return json.loads(content.split(_SMALL30_MARKER+"\n```json\n",1)[1].split("\n```",1)[0])


def _small30_write(report,data):
    from statistics import mean,median
    content=report.read_text(encoding='utf-8')
    prefix=content.split('\n\n## SMALL_30_39复核',1)[0]
    text='\n\n## SMALL_30_39复核\n\n'
    text+=('本章独立于上面的历史N26/55/85实验，不修改其SMALL标签或24条原始记录，不混合计算平均值。'
           '只选V2_MODEL_DEVELOPMENT_CONSUMED；目录名不定义角色，所选两个data/ID_TEST旧路径已明确属于消耗过的development，非ID_TEST_SEALED。'
           '未生成样本、未删除焊缝、未改角色或TRAIN/DEV划分。By=[5.5,6.5]仍是任务层假设，非URDF/IK验证。\n\n')
    text+=f"已检查 {data['inspected_development_workbooks']} 个development workbook的META与实际行数；找到{data['eligible_sheet_count']}个N30–39 sheet，来自{data['eligible_workbook_count']}个workbook；距35最近的候选{len(data['candidates'])}个。\n\n"
    text+='选择规则：'+data['selection_rule']+'。清单在求解前固定，三例实际N均为35。\n\n'
    text+=_range_table(['编号','角色','workbook','sheet','N','总长度m','X/Y patterns','上/下/共享/mandatory','cross y6','x_up/x_low'],[
        [e['selection_ordinal'],e['geometry_role'],e['relative_path'],e['sheet_name'],e['N'],e['total_length_m'],
         f"{e['x_patterns']}/{e['y_patterns']}",'/'.join(str(e[k]) for k in ('upper_only','lower_only','both','mandatory_y')),
         e['cross_y6'],f"{e['x_up']:.4f}/{e['x_low']:.4f}"] for e in data['entries']])
    text+=('\n\nA=0.2/0.2，D=0.5/0.5；seeds 20261081/20261082/20261083。其余沿用SA_OI_ALNS_V2、M192=144+48、Kdp8/Kref2/Kref_total4、'
           'heuristic C2/C4、TWO_OPT_STAR OFF、construction5/kinit5/bootstrap1、B32与既有certifier。18个真实60秒wall-clock run；'
           'worker各绑定独立物理核，A/D顺序交替。观察记录开销计入预算。@60独立认证，绝不以overshoot final Cmax补填。\n\n'
           'first_certified_initial为初始化中首个通过独立认证的reference结果；chosen_initial为初始化完成后实际选中的解。'
           'iterations/accepted/valid/funnel取60秒内完成的最后一个iteration boundary；reference_status_counts_at60按实际认证完成时刻≤60计数，可能包含尚未完成的最后一轮ref。'
           'global-best另保留原生时间戳，次级目标改善也可触发update；同时报告Cmax是否真正下降。四机器人负载按R0/R1/R2/R3排序，含各child独立setup/weld/post。\n\n')
    rows=data['runs'];text+=f"执行完成 {len(rows)}/18。\n\n"
    if rows:
        text+=_range_table(['例','seed','组','认证','首个init Cmax','选中init Cmax','Cmax@60','init s','iter≤60','accept≤60','best≤60','ref≤60','valid率','WHOLE/Y/X','四robot process load s','负载差s','空行走s','等待s','runtime/overshoot s'],[
            [r['instance'],r['seed'],r['group'],r.get('certified_at60'),
             (r.get('first_certified_initial') or {}).get('cmax'),(r.get('chosen_initial') or {}).get('cmax'),r.get('cmax_at60'),r.get('init_time_s'),
             r.get('counters_at60',{}).get('iterations'),r.get('counters_at60',{}).get('accepted_moves'),r.get('counters_at60',{}).get('global_best_updates'),
             r.get('reference_calls_at60'),r.get('candidate_valid_rate_at60'),
             '/'.join(str((r.get('at60') or {}).get('patterns',{}).get(k,0)) for k in ('WHOLE','Y_SPLIT','X_SPLIT')),
             '/'.join(f"{v:.2f}" for v in (r.get('at60') or {}).get('process_loads',[])),
             (r.get('at60') or {}).get('process_imbalance'),(r.get('at60') or {}).get('empty_travel_s'),(r.get('at60') or {}).get('waiting_s'),
             f"{r.get('elapsed_s',0):.3f}/{r.get('overshoot_s',0):.3f}"] for r in sorted(rows,key=lambda r:(r['instance'],r['seed'],r['group']))])
        index={(r['instance'],r['seed'],r['group']):r for r in rows}
        pairs=[]
        for e in data['entries']:
            for seed in _SMALL30_SEEDS:
                a=index.get((e['selection_ordinal'],seed,'A'));d=index.get((e['selection_ordinal'],seed,'D'))
                if a and d and a.get('certified_at60') and d.get('certified_at60'):
                    ai=a['chosen_initial']['cmax'];di=d['chosen_initial']['cmax']
                    pairs.append([e['selection_ordinal'],seed,d['cmax_at60']-a['cmax_at60'],
                        100*(d['cmax_at60']/a['cmax_at60']-1),di-ai,100*(di/ai-1),
                        ai-a['cmax_at60'],di-d['cmax_at60']])
        text+='\n\n配对差D−A；负值表示D更好。搜索改善为选中init−@60，正值表示ALNS降低Cmax。\n\n'
        text+=_range_table(['例','seed','@60绝对差s','@60相对差%','init绝对差s','init相对差%','A搜索改善s','D搜索改善s'],pairs)
        summary=[]
        for label,items in [('ALL',pairs)]+[(f"S{i}",[p for p in pairs if p[0]==i]) for i in (1,2,3)]:
            if not items: continue
            summary.append([label,len(items),mean(p[2] for p in items),median(p[2] for p in items),
                mean(p[3] for p in items),median(p[3] for p in items),sum(p[2]<-1e-8 for p in items),
                sum(abs(p[2])<=1e-8 for p in items),sum(p[2]>1e-8 for p in items)])
        text+='\n\n'+_range_table(['范围','配对数','绝对差均值s','绝对差中位s','相对差均值%','相对差中位%','改善','持平','恶化'],summary)
    text+='\n\n'+data.get('conclusions','全部18次完成后补充解释与下一步结论。')
    text+='\n\n### I3历史边界案例的定向诊断\n\n'+data.get('i3_analysis','保留历史恶化发现；仅补初始化及旧解重建诊断，不重跑历史整组60秒实验。')
    text+='\n\n完整pytest：'+data.get('tests','待执行')+'。\n\n### SMALL_30_39逐次原始记录\n\n'
    text+=_SMALL30_MARKER+'\n```json\n'+json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n```\n'
    report.write_text(prefix+text,encoding='utf-8')


def _small30_main(action,workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from mrta_ranker.dataset import physical_core_affinities
    root=Path(__file__).resolve().parents[1];ppo=root.parent/'ppo'
    report=root/'docs/YX_SPLIT_RANGE_05_CHANGE.md'
    content=report.read_text(encoding='utf-8')
    data=_small30_read(report) if _SMALL30_MARKER in content else _small30_select(root,ppo)
    roles=json.loads((root/'data/manifests/PPO_V2_DATA_ROLES_V1.json').read_text(encoding='utf-8'))
    assert_v2_solver_access_allowed(roles,[e['relative_path'] for e in data['entries']],
        allowed_roles=('V2_MODEL_DEVELOPMENT_CONSUMED',))
    if action=='i3' and not data['i3_diagnostics']:
        data['i3_diagnostics']=_small30_i3(root,ppo)
    _small30_write(report,data)
    if action!='run':
        print(f"SMALL_30_39: {len(data['entries'])} selected, actual N {[e['N'] for e in data['entries']]}; {len(data['runs'])}/18 complete.",flush=True)
        if action=='i3': print(json.dumps(data['i3_diagnostics'],ensure_ascii=False),flush=True)
        return 0
    done={(r['instance'],r['seed'],r['group']) for r in data['runs']}
    masks=physical_core_affinities()[:workers];jobs=[]
    for i,e in enumerate(data['entries']):
        for j,seed in enumerate(_SMALL30_SEEDS):
            groups=['A','D'] if (i+j)%2==0 else ['D','A']
            for group in groups:
                if (e['selection_ordinal'],seed,group) not in done:
                    jobs.append((e,seed,group,str(ppo),0))
    import multiprocessing
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    for mask in masks: queue.put(mask)
    with ProcessPoolExecutor(max_workers=len(masks),mp_context=context,initializer=_range_worker_init,initargs=(queue,)) as pool:
        futures={pool.submit(_small30_worker,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try: row=future.result()
            except Exception as exc:
                row=dict(instance=job[0]['selection_ordinal'],seed=job[1],group=job[2],certified_at60=False,error=repr(exc))
            data['runs'].append(row);_small30_write(report,data)
            print(f"small30 {len(data['runs'])}/18: S{row['instance']} seed={row['seed']} {row['group']} certified={row['certified_at60']} Cmax@60={row.get('cmax_at60')}",flush=True)
    return 0 if len(data['runs'])==18 and all(r.get('certified_at60') for r in data['runs']) else 2


def main() -> int:
    import sys
    if "--small30" in sys.argv:
        current=argparse.ArgumentParser(description="N30-39 development-only range recheck")
        current.add_argument("--small30",choices=("select","run","report","i3"),required=True)
        current.add_argument("--workers",type=int,default=4)
        args=current.parse_args()
        return _small30_main(args.small30,args.workers)
    if "--yx-range" in sys.argv:
        current = argparse.ArgumentParser(description="Finite Y/X range development sensitivity")
        current.add_argument("--yx-range", choices=("catalog", "run", "report"), required=True)
        current.add_argument("--workers", type=int, default=4)
        args = current.parse_args()
        return _range_main(args.yx_range, args.workers)
    parser = argparse.ArgumentParser(description="Run Phase 3 compatibility smoke on frozen PPO instances")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument(
        "--manifest", default="data/manifests/PPO_DATASET_MANIFEST_V1.json"
    )
    parser.add_argument(
        "--smokeset", default="data/manifests/PPO_PHASE3_SMOKESET_V2.json"
    )
    parser.add_argument("--include-30", action="store_true")
    parser.add_argument(
        "--source-commit-label",
        required=True,
        help="Explicit source revision/tree label for this local result",
    )
    args = parser.parse_args()
    if not args.source_commit_label:
        parser.error("--source-commit-label is required for historical smoke")
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    root = Path(args.ppo_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    smokeset = json.loads(Path(args.smokeset).read_text(encoding="utf-8"))
    by_id = {entry["instance_id"]: entry for entry in manifest["instances"]}
    selected = []
    for selected_entry in smokeset["instances"]:
        entry = dict(by_id[selected_entry["instance_id"]])
        entry["tier"] = selected_entry["tier"]
        entry["dataset_manifest_hash"] = manifest["dataset_manifest_hash"]
        selected.append(entry)
    results = [_run(entry, root, 5.0, args.source_commit_label) for entry in selected]
    if args.include_30:
        results.extend(
            _run(entry, root, 30.0, args.source_commit_label)
            for entry in selected
            if entry["tier"] in {"medium", "large"}
        )
    payload = {
        "smokeset_id": smokeset["smokeset_id"],
        "dataset_manifest_id": manifest["dataset_manifest_id"],
        "dataset_manifest_hash": manifest["dataset_manifest_hash"],
        "formal_scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "formal_scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "deadlock_rollout_budget": FORMAL_SCOPE_V1_1.deadlock_rollout_budget,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["final_certification"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
