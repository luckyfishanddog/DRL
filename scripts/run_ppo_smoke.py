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
    entry,seed,group,ppo,_=job[:5]
    cfg=_range_config(group);parents=_range_parents(entry,Path(ppo))
    from dataclasses import replace
    search=_range_search_config()
    if len(job)>5:
        search=replace(search,initialization_clearance_aware=job[5])
    with _Small30Telemetry() as trace:
        result=run_sa_oi_alns_v2(parents,cfg,search,seed=seed,
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
        init_status_counts=dict(stats.init_status_counts),init_reference_calls=stats.init_reference_calls,
        construction_attempts=stats.construction_attempts,
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



# Current .5-domain diagnosis. The existing historical range entry points stay unchanged.
def _init_screening_entries(root):
    small = _small30_read(root/'docs/YX_SPLIT_RANGE_05_CHANGE.md')['entries']
    old = _range_entries(root)
    # N55: lowest existing static selection ordinal; never selected by Cmax.
    entries = [dict(next(e for e in old if e['selection_ordinal']==3), label='I3'),
               dict(small[0], label='S1'), dict(small[1], label='S2'),
               dict(min((e for e in old if e['tier'].upper()=='MEDIUM'),
                        key=lambda e:e['selection_ordinal']), label='N55')]
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    roles=json.loads((root/'data/manifests/PPO_V2_DATA_ROLES_V1.json').read_text(encoding='utf-8'))
    assert_v2_solver_access_allowed(roles,[e['relative_path'] for e in entries],
        allowed_roles=('V2_MODEL_DEVELOPMENT_CONSUMED',))
    return entries


class _S1PoolCapture:
    """Capture actual native pools after completed iterations 2 and 8, before mutation."""
    def __init__(self, seed):
        self.seed=seed; self.boundary=None; self.pending=None; self.states=[]
    def __call__(self,event,value):
        import pickle
        if event=='boundary':
            self.boundary=(pickle.loads(pickle.dumps(value)) if value['iteration'] in (2,8) else dict(value))
        elif event=='selection':
            if self.boundary['iteration'] not in (2,8) or self.boundary['elapsed']>60.:
                self.pending=None; return
            complete,c2=value
            context=dict(self.boundary,seed=self.seed,
                generation_seed=self.seed+self.boundary['iteration']*65537,
                rule='current accepted state after completed iterations 2 and 8; next native pool',
                adaptive_snapshot_phase='completed iteration boundary, before generation')
            self.pending=dict(context=pickle.dumps(context),pool=pickle.dumps(complete),
                c2=[c.solution.canonical_hash for c in c2])
        elif event=='iteration_evaluation' and self.pending is not None:
            c3,c4,_=value
            self.pending['c3']=pickle.dumps(c3)
            self.pending['c4']=[r.direction_candidate.screened.solution.canonical_hash for r in c4]
            self.pending['online_c4']=[dict(identity=r.direction_candidate.screened.solution.canonical_hash,
                status=r.schedule.status.value,cmax=None if r.metrics is None else r.metrics.cmax)
                for r in c4]
            self.states.append(self.pending);self.pending=None


def _init_screening_capture(job):
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry,seed,ppo=job;cfg=_range_config('D');parents=_range_parents(entry,Path(ppo))
    capture=_S1PoolCapture(seed)
    with _Small30Telemetry() as trace:
        def observe(event,value):
            trace(event,value);capture(event,value)
        result=run_sa_oi_alns_v2(parents,cfg,_range_search_config(),seed=seed,
            source_commit='LOCAL_S1_NATIVE_POOL_DIAGNOSIS',allow_unverified_source=True,
            observer=observe,ranker=None)
    return dict(seed=seed,states=capture.states,runtime=result.runtime,
        cmax_at60=result.anytime[60.]['cmax'],iterations=result.stats.iterations,
        online_reference_calls=len(trace.references),online_statuses=dict(result.stats.init_status_counts),
        actual_N=len(parents),core_mask=_RANGE_CORE_MASK)


def _screening_loss_summary(rows,current_cmax):
    """All gains use this accepted state's Cmax, never global best."""
    from collections import Counter
    def group(selected):
        feasible=[r for r in selected if r['status']=='FEASIBLE_CERTIFIED']
        positive=[r for r in feasible if (current_cmax-r['cmax'])/current_cmax>=.005]
        return dict(count=len(selected),statuses=dict(Counter(r['status'] for r in selected)),
            material_count=len(positive),best_cmax=min((r['cmax'] for r in feasible),default=None),
            best_gain=max(((current_cmax-r['cmax'])/current_cmax for r in feasible),default=None))
    pool=group(rows); c2=group([r for r in rows if r['c2']])
    direction=group([r for r in rows if r['c2'] and r['direction_feasible']])
    c4=group([r for r in rows if r['c4']])
    if pool['material_count']==0:
        loss='generation/reference (no certified material candidate in captured pool)'
    elif c2['material_count']==0: loss='C2 screening'
    elif c4['material_count']==0: loss='C4 screening'
    else: loss='captured C4 retains material improvement'
    return dict(current_cmax=current_cmax,pool=pool,c2=c2,direction=direction,c4=c4,
        material_lost_c2=pool['material_count']-c2['material_count'],
        material_lost_c4=direction['material_count']-c4['material_count'],primary_loss=loss)


def _init_screening_label(job):
    import pickle,time
    from mrta_search.direction import optimize_directions_with_initial_feasibility
    from mrta_search.pipeline import decision_family
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    state_id,context_blob,pool_blob,c2,c4=job
    context=pickle.loads(context_blob);pool=pickle.loads(pool_blob);cfg=_range_config('D')
    state_cert=certify_schedule(context['solution'],context['schedule'],cfg,scope=FORMAL_SCOPE_V2)
    if not state_cert.certified: raise ValueError(state_cert.errors)
    evaluator=FormalReferenceEvaluator(FORMAL_SCOPE_V2);rows=[];started=time.perf_counter();calls=0
    for index,candidate in enumerate(pool):
        sol=candidate.solution;identity=sol.canonical_hash
        tick=time.perf_counter();direction=optimize_directions_with_initial_feasibility(sol,cfg)
        dp_s=time.perf_counter()-tick
        row=dict(index=index,identity=identity,c2=identity in c2,c4=identity in c4,
            family=decision_family(candidate),direction_feasible=direction.total_empty_travel is not None,
            direction_status=direction.status.value,directions=direction.directions,direction_s=dp_s,status='DIRECTION_INFEASIBLE',
            cmax=None,reference_s=0.,certifier_s=0.,diagnostics=list(direction.diagnostics))
        if row['direction_feasible']:
            tick=time.perf_counter();calls+=1
            try:
                schedule=evaluator(sol,cfg,orientations={r:direction.directions[r] for r in range(4)})
                row.update(status=schedule.status.value,diagnostics=list(schedule.diagnostics))
                row['reference_s']=time.perf_counter()-tick
                if schedule.status.value=='FEASIBLE':
                    tick=time.perf_counter();cert=certify_schedule(sol,schedule,cfg,scope=FORMAL_SCOPE_V2)
                    row['certifier_s']=time.perf_counter()-tick
                    row.update(status='FEASIBLE_CERTIFIED' if cert.certified else 'NUMERIC_FAILURE',
                        cmax=schedule.cmax if cert.certified else None,certification_errors=list(cert.errors))
            except (ArithmeticError,OverflowError,ValueError) as exc:
                row.update(status='NUMERIC_FAILURE',diagnostics=[str(exc)],reference_s=time.perf_counter()-tick)
        row['gain']=None if row['cmax'] is None else (context['metrics'].cmax-row['cmax'])/context['metrics'].cmax
        rows.append(row)
    return dict(state_id=state_id,rows=rows,offline_reference_calls=calls,offline_elapsed_s=time.perf_counter()-started,
        summary=_screening_loss_summary(rows,context['metrics'].cmax))



def _init_screening_init(job):
    from dataclasses import replace
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry,variant,ppo=job;cfg=_range_config('D');parents=_range_parents(entry,Path(ppo))
    search=replace(_range_search_config(budget=None,iterations=0),
        initialization_clearance_aware=variant=='IMPROVED')
    with _Small30Telemetry() as trace:
        result=run_sa_oi_alns_v2(parents,cfg,search,seed=20261101,
            source_commit='LOCAL_INIT_EQUAL_BUDGET_DIAGNOSIS',allow_unverified_source=True,observer=trace)
    init=result.initialization
    details=None if init.solution is None else _small30_solution_details(
        init.solution,init.schedule,result.best_metrics,cfg)
    return dict(instance=entry['label'],variant=variant,N=len(parents),workbook=entry['relative_path'],
        sheet=entry['sheet_name'],certified=bool(result.final_certification and result.final_certification.certified),
        first_certified=trace.first_init,chosen=details,strategy=init.winning_strategy,
        fallback=init.winning_strategy=='RAIL_SERIAL_BOOTSTRAP',init_time_s=result.stats.init_time,
        total_elapsed_s=result.runtime,reference_calls=result.stats.init_reference_calls,
        construction_attempts=result.stats.construction_attempts,core_mask=_RANGE_CORE_MASK,
        references=trace.references,attempts=[dict(index=a.construction_index,strategy=a.strategy,
            duplicate=a.duplicate,direction_status=None if a.directions is None else a.directions.status.value,
            status=None if a.schedule is None else a.schedule.status.value,
            cmax=None if a.schedule is None else a.schedule.cmax,diagnostics=list(a.diagnostics),
            certified=bool(a.certification and a.certification.certified)) for a in init.attempts])


def _init_screening_run(job):
    entry,seed,variant,ppo=job
    row=_small30_worker((entry,seed,'D',ppo,0,variant=='IMPROVED'))
    row.update(instance=entry['label'],variant=variant,tier='INITIALIZATION_DEVELOPMENT')
    return row



def _init_screening_search_lines(db):
    from statistics import mean,median
    runs=[json.loads(r[0]) for r in db.execute("SELECT result FROM results WHERE kind='search60' ORDER BY id")]
    decision=next((json.loads(r[0]) for r in db.execute("SELECT result FROM results WHERE kind='decision' AND id='production'")),{})
    lines=['## 16次真实60秒对比','',
        '零搜索可行性与预算全部通过，因此保留S2退化风险，继续按既定四实例×两个新seed×两个variant实际运行。没有用旧结果代替BASELINE。每个worker绑定独立物理核；两个variant按实例/seed交替提交，均使用单进程原生60秒预算。初始化计入60秒；唯一算法差异是第五初始化构造。',
        f'实际完成{len(runs)}/16。采用native best-event截止60秒的Cmax，所选解运行后再次独立认证，overshoot final值不用于比较。严格Cmax更新数排除等Cmax次级指标更新。16次完整逐run结果保存在SQLite results/search60。','',
        '|实例|seed|variant|初始化Cmax|Cmax@60|搜索下降s|初始化s|完整迭代≤60|accepted≤60|严格Cmax更新≤60|valid candidate比例|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        events=[float(c) for t,c in r['best_events'] if t<=60.]
        strict=sum(b<a-1e-9 for a,b in zip(events,events[1:]))
        count=r['counters_at60'];initial_cmax=r['chosen_initial']['cmax']
        lines.append(f"|{r['instance']}|{r['seed']}|{r['variant']}|{initial_cmax:.2f}|{r['cmax_at60']:.2f}|{initial_cmax-r['cmax_at60']:.2f}|{r['init_time_s']:.3f}|{count['iterations']}|{count['accepted_moves']}|{strict}|{r['candidate_valid_rate_at60']*100:.2f}%|")
    lines += ['', '|实例|seed|variant|@60四机器人process load s|负载差s|init ref / 总ref≤60|FEASIBLE/DEADLOCK/INFEASIBLE/NUMERIC≤60|认证@60|runtime / overshoot s|',
        '|---|---:|---|---|---:|---:|---|---|---|']
    for r in runs:
        q=r['at60'];loads='/'.join(f'{x:.2f}' for x in q['process_loads']);statuses=r['reference_status_counts_at60']
        counts='/'.join(str(statuses.get(k,0)) for k in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE'))
        lines.append(f"|{r['instance']}|{r['seed']}|{r['variant']}|{loads}|{q['process_imbalance']:.2f}|{r['init_reference_calls']} / {r['reference_calls_at60']}|{counts}|{'是' if r['certified_at60'] else '否'}|{r['elapsed_s']:.3f} / {r['overshoot_s']:.3f}|")
    lines += ['', '|实例|seed|BASELINE Cmax@60|IMPROVED Cmax@60|绝对差s|相对差|', '|---|---:|---:|---:|---:|---:|']
    paired=[]
    for label in ('I3','S1','S2','N55'):
        for seed in (20261101,20261102):
            a=next((r for r in runs if r['instance']==label and r['seed']==seed and r['variant']=='BASELINE'),None)
            b=next((r for r in runs if r['instance']==label and r['seed']==seed and r['variant']=='IMPROVED'),None)
            if a is None or b is None:continue
            delta=b['cmax_at60']-a['cmax_at60'];relative=delta/a['cmax_at60']*100.
            paired.append(dict(instance=label,seed=seed,delta=delta,relative=relative))
            lines.append(f"|{label}|{seed}|{a['cmax_at60']:.2f}|{b['cmax_at60']:.2f}|{delta:+.2f}|{relative:+.2f}%|")
    if paired:
        wins=sum(r['delta'] < -1e-8 for r in paired);losses=sum(r['delta'] > 1e-8 for r in paired)
        lines += ['',f"全部{len(paired)}对：{wins}改善 / {len(paired)-wins-losses}持平 / {losses}恶化；绝对差均值{mean(r['delta'] for r in paired):+.2f}s、中位数{median(r['delta'] for r in paired):+.2f}s；相对差均值{mean(r['relative'] for r in paired):+.2f}%、中位数{median(r['relative'] for r in paired):+.2f}%。这些规模不能混称为同一个SMALL结论。",'',
            '|实例|两个seed相对差均值|两个seed绝对差均值s|','|---|---:|---:|']
        for label in ('I3','S1','S2','N55'):
            rows=[r for r in paired if r['instance']==label]
            if rows:lines.append(f"|{label}|{mean(r['relative'] for r in rows):+.2f}%|{mean(r['delta'] for r in rows):+.2f}|")
    lines += ['', '生产决定：'+decision.get('reason','等待全部16次结果，不提前推荐策略。'),'',
        'S1改进初始化两个seed均完成19次迭代、global-best更新=0、C4认证=0，@60 Cmax等于初始化Cmax。其reference（含初始化）各有41个DEADLOCK/1个FEASIBLE，不能将初始化下降当作ALNS搜索提升；未对这些新trajectory状态另做全池标注，不能据选中项推断全池无好候选。',
        'S1原初始化的固定池诊断与初始化对比是两个问题：新初始化带来的trajectory变化不改变已测得的C2/C4漏选事实，不能把初始化收益包装为排序器已修复。','']
    return lines


def _init_screening_report(root,db):
    import pickle
    from collections import Counter
    def results(kind):
        return [json.loads(r[0]) for r in db.execute('SELECT result FROM results WHERE kind=? ORDER BY id',(kind,))]
    screenings=results('screening');initial=results('initialization');trajectories=results('trajectory')
    labels=[json.loads(r[0]) for r in db.execute('SELECT result FROM labels')]
    entries=_init_screening_entries(root)
    validation=results('validation')[0]
    lines=['# 初始化鲁棒性与S1筛选诊断（0.5 m开发域）','',
        '结论：S1主要在C2流失真实改善，C4也存在独立漏选。空间分界初始化原型修复了I3串行退化并改善S1初始化，但使S2初始化恶化43.95%，后续已执行16次真实60秒配对。heuristic C2/C4保持原版；生产初始化决定见后文。','',
        '## 实际改动','',
        '- 在既有第五个RAIL_MONOTONE_BALANCED构造内加入可选空间分界：若存在满足左右任务X包络间距≥interference_dx的非空分区，选择其中加工负载最均衡者；否则沿用原分界。共享焊缝上下轨分配、WHOLE/mandatory Y选择及方向算法均保持原规则。',
        '- `SearchConfig.initialization_clearance_aware=False`为默认；显式True才启用CLEARANCE_BALANCED_BOOTSTRAP。保留原版消融。没有第六个正常构造，没有额外reference上限。',
        '- 复用run_ppo_smoke.py的`--init-screening capture/label/init/run/report`入口；固定状态、完整候选、标签、零搜索结果保存在唯一SQLite。未新建runner、protocol、manifest或hash系统。',
        '- 全部采用delta_x=delta_y=0.50、最短子段0.20、干涉距离0.50；M192=144+48、Kdp8/Kref2/Kref_total4、原SA、原family policy、TWO_OPT_STAR OFF、B32及独立certifier。共享覆盖[5.5,6.5]仍是任务层假设，未进行URDF/IK验证。','',
        '## 固定实例与状态规则','',
        '全部经既有角色清单检查，仅使用V2_MODEL_DEVELOPMENT_CONSUMED。物理目录名ID_TEST不代表角色为封存测试；S1/S2所在workbook在清单中的角色确实为已消耗development。N55按既有静态选择序号最小的MEDIUM固定，未读取Cmax挑选。','',
        '|实例|实际N|workbook|sheet|','|---|---:|---|---|']
    for e in entries:
        n=26 if e['label']=='I3' else 55 if e['label']=='N55' else 35
        lines.append(f"|{e['label']}|{n}|{e['relative_path']}|{e['sheet_name']}|")
    lines += ['', 'S1使用20261081、20261083，各在第2、第8次已完成迭代后的current accepted state，捕获紧接着的原生生成/筛选迭代（按1起算为第3、第9次）。选择不依赖Cmax或候选标签。保存了实际solution、directions、schedule、current Cmax、seed、iteration、生成seed、SA随机状态、温度、自适应权重/记录、完整valid unique pool及C2/C3/C4。原始捕获的adaptive快照位于selection时，已明确记录；生成只追加LNS选择历史，权重与boundary时一致。后续observer实现已在boundary冻结对象。全池来自真实运行，没有由历史最终解推断或伪造重放。', '',
        '|seed|完成迭代|状态时间s|current Cmax|valid unique|pool最优改善|C2最优改善|direction通过|direction后最优改善|C4最优改善|material数量 pool/C2/C4|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def gain(value):return '无认证候选' if value is None else f'{value*100:.2f}%'
    for result in screenings:
        sid=result['state_id'];ctx=pickle.loads(db.execute('SELECT context FROM states WHERE id=?',(sid,)).fetchone()[0]);q=result['summary']
        lines.append(f"|{ctx['seed']}|{ctx['iteration']}|{ctx['elapsed']:.3f}|{q['current_cmax']:.2f}|{q['pool']['count']}|{gain(q['pool']['best_gain'])}|{gain(q['c2']['best_gain'])}|{q['direction']['count']}/8|{gain(q['direction']['best_gain'])}|{gain(q['c4']['best_gain'])}|{q['pool']['material_count']}/{q['c2']['material_count']}/{q['c4']['material_count']}|")
    total=Counter(r['status'] for r in labels);c2=[r for r in labels if r['c2']];c4=[r for r in labels if r['c4']]
    lines += ['', '## S1各阶段的真实损失','',
        'material improvement严格按该current accepted state的(Cs−Cc)/Cs≥0.005，未混用global-best。全池上限是当前direction+B32产生并独立认证的上限，不是数学最优；DEADLOCK仅表示该有界调度没有找到解。','',
        '|阶段|候选数|认证FEASIBLE|DEADLOCK|direction-infeasible|INFEASIBLE|NUMERIC_FAILURE|material|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,rows in [('完整valid unique pool',labels),('C2 Top8',c2),('direction-feasible C2',[r for r in c2 if r['direction_feasible']]),('C4 Top2',c4)]:
        count=Counter(r['status'] for r in rows);material=sum(r['gain'] is not None and r['gain']>=.005 for r in rows)
        lines.append(f"|{name}|{len(rows)}|{count['FEASIBLE_CERTIFIED']}|{count['DEADLOCK']}|{count['DIRECTION_INFEASIBLE']}|{count['INFEASIBLE']}|{count['NUMERIC_FAILURE']}|{material}|")
    lines += ['',
        '- Generation：四个池均有8–15个认证material候选，最佳改善9.01%–11.96%。本次四状态的失败不能归因于生成器完全没有好候选，也不能据此保证整条trajectory始终有好候选。',
        '- C2：44个material仅保留3个，漏掉41个（93.18%）。20261081两个状态全部漏掉；20261083两个状态保留了全池最优候选。按候选数量capture为6.82%，这与按最佳改善幅度衡量的capture不同。',
        '- Direction：32个C2候选中29个通过，3个direction-infeasible不进入C4。C2认证候选的改善上限与direction-feasible C2上限相同；这3个不能记为C4排序损失。',
        '- C4：C2保留的3个material只选中1个。20261083早期状态丢掉2个，包括全池最佳10.13%改善；中期状态保住11.96%改善。其余两个状态C2已经没有material，不向C4归责。按条件候选数量capture为33.33%。',
        '- Reference：direction-feasible全池297/422认证（70.38%），C2中5/29（17.24%），C4中1/8（12.50%）；C4其余7个均DEADLOCK。这说明筛选后的集合强烈偏向当前B32难以排出的候选，不是整个池都没有可行改善。C2的24个DEADLOCK诊断共出现31条同轨机器人阻塞关系；常涉及WELD/POST停留及MOVE冲突，没有把DEADLOCK称为绝对不可行，也未修改调度器。',
        '- 结构：C2的DEADLOCK含16个STRUCTURAL、4个TARGET_Y、3个TARGET_WHOLE、1个TARGET_X；C4的DEADLOCK含4个STRUCTURAL、2个TARGET_WHOLE、1个TARGET_X。损失不限于新增外侧拆分。',
        '- 四个池的8个C4候选在线与离线状态、Cmax逐项完全一致。离线原始标签独立计时/计数，未使用旧0.2域标签。','',
        '|固定状态|离线reference calls|离线wall-clock s（各工作进程）|', '|---|---:|---:|']
    for r in screenings:lines.append(f"|{r['state_id']}|{r['offline_reference_calls']}|{r['offline_elapsed_s']:.3f}|")
    lines += ['',f"离线合计{sum(r['offline_reference_calls'] for r in screenings)}次reference；进程耗时之和{sum(r['offline_elapsed_s'] for r in screenings):.3f}s，四池并行，不能解释为单进程总墙钟。在线轨迹另行计数如下；其Cmax只用于说明本次真实捕获运行，不与旧历史结果充当配对比较。真实wall-clock截止会使完成迭代数变化，不能保证与旧60秒终点相同。",'',
        '|seed|本次Cmax@60|总runtime s|overshoot s|总完成迭代（含overshoot）|在线reference calls（含初始化与overshoot）|',
        '|---:|---:|---:|---:|---:|---:|']
    for r in trajectories:lines.append(f"|{r['seed']}|{r['cmax_at60']:.2f}|{r['runtime']:.3f}|{max(0,r['runtime']-60):.3f}|{r['iterations']}|{r['online_reference_calls']}|")
    lines += ['', '## 初始化：实际失败原因与零搜索比较','',
        '另有4次原版零搜索失败原因诊断，合计24次初始化reference，独立保存在results/initial_failure；不混入422次全池离线标注、8次初始化对比或16次60秒配对的计数。',
        '原版I3：LOAD_FIRST、X_ORDER_AWARE及RAIL_MONOTONE均被B32判DEADLOCK；RAIL_BALANCED/SPATIAL_SPREAD在首任务同轨顺序检查失败。单调构造下轨左右X包络重叠0.712m，方向备用尝试仍失败，最后R0/R2串行fallback认证。S1也有首任务顺序失败与同轨阻塞；仅按四机器人加工负载平衡不足以保证有可排出的路线。','',
        '每个variant仍最多5个正常构造，Kinit_ref=5；唯一第五构造DEADLOCK时仍最多4次原有有界方向reference尝试，串行fallback仍最多1次。所以“5”不是总reference硬上限；原规则总上界是5+4+1。此次没有增加任何上限，失败构造/重复构造按原规则跳过。所有接受解经过canonicalization、direction、B32及独立certifier。零搜索使用同一组实例和固定seed20261101（构造不读取seed），初始化耗时来自原计时器，首认证时间从run开始计。','',
        '|实例|variant|认证|选中Cmax|四机器人process load s|负载差s|串行fallback|reference calls|首认证s|初始化s|总runtime s|',
        '|---|---|---|---:|---|---:|---|---:|---:|---:|---:|']
    for r in initial:
        q=r['chosen'];loads='/'.join(f'{x:.2f}' for x in q['process_loads'])
        lines.append(f"|{r['instance']}|{r['variant']}|{'是' if r['certified'] else '否'}|{q['cmax']:.2f}|{loads}|{q['process_imbalance']:.2f}|{'是' if r['fallback'] else '否'}|{r['reference_calls']}|{r['first_certified']['elapsed']:.3f}|{r['init_time_s']:.3f}|{r['total_elapsed_s']:.3f}|")
    lines += ['', '|实例|初始化Cmax配对变化(IMPROVED−BASELINE)|相对变化|', '|---|---:|---:|']
    for label in ('I3','S1','S2','N55'):
        a=next(r for r in initial if r['instance']==label and r['variant']=='BASELINE')['chosen']['cmax']
        b=next(r for r in initial if r['instance']==label and r['variant']=='IMPROVED')['chosen']['cmax']
        lines.append(f'|{label}|{b-a:+.2f}s|{(b/a-1)*100:+.2f}%|')
    lines += ['',
        '8/8零搜索结果认证、NUMERIC_FAILURE=0、独立认证错误=0。I3由只有R0/R2工作变为四机器人工作，Cmax改善39.38%；S1改善23.23%；N55 Cmax相同。但S2为取得空间间隔，使上轨负载分成893.18/2849.41，负载差从728.23增至2042.87，Cmax恶化43.95%。这是空间分界与加工均衡之间的真实权衡，需要检验60秒搜索能否补偿，不是通过增加reference预算获得胜利。',
        '新构造完全由当前几何和当前0.5合法WHOLE/Y候选独立生成；没有读历史A优秀解或按实例ID/seed特判。历史A重建结果仅保留为旧证据，不作为本次初始化输入。', '',
        *_init_screening_search_lines(db),
        '## 测试与明确下一步','',
        f'完整命令：`D:\\pybullet_test\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider`（DRL项目目录）。**{validation["passed"]} passed in {validation["elapsed_s"]:.2f}s**。覆盖原五槽位/预算、几何独立构造与重复性、0.5目录/mandatory Y/0.2长度、独立认证拒绝不得接受、固定iteration捕获、current-state material边界、direction与numeric不误归责C4、在线/离线调用隔离；既有X固定机器人、同轨/连续干涉与历史数值回归全部通过。','',
        '没有发现需要绕过或修改正式合法性/调度约束的代码错误。发现的是初始化空间分界策略弱点和C2/C4启发式目标失配；新的prototype也暴露只追求空间分离的负载代价。',
        '下一步优先做新0.5域C2筛选改进，重点检验如何在原M192/Kdp8内保留认证改善并减少同轨阻塞倾向；C4随后处理已通过C2的漏选。当前不需要扩大候选池或增加operator。初始化后续仍需同时约束空间冲突与负载代价，推荐决定以本轮配对结果为依据。',
        '四个状态已提供“有真实好候选而排序漏掉”的必要证据，支持将来有针对性地采集0.5域C2/C4标签；覆盖范围仍不足以直接启动大规模训练。本轮未采集新的MLP训练集、未训练MLP/GAT，旧14,893标签与所有历史实验未改动。','']
    (root/'docs/INITIALIZATION_AND_S1_SCREENING_FIX.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report complete: S1 screening and actual initialization/search comparisons.',flush=True)

def _init_screening_main(action,workers):
    import sqlite3,pickle,multiprocessing
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from mrta_ranker.dataset import physical_core_affinities
    root=Path(__file__).resolve().parents[1];ppo=root.parent/'ppo'
    entries=_init_screening_entries(root)
    db=sqlite3.connect(root/'data/development/s1_screening_diagnostic.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS states (id TEXT PRIMARY KEY, context BLOB, pool BLOB, c3 BLOB, c2 TEXT, c4 TEXT, online_c4 TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS labels (state_id TEXT, candidate_index INTEGER, result TEXT, PRIMARY KEY(state_id,candidate_index))')
    db.execute('CREATE TABLE IF NOT EXISTS results (kind TEXT, id TEXT, result TEXT, PRIMARY KEY(kind,id))')
    db.commit()
    def save(kind,key,value):
        db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?)',(kind,str(key),json.dumps(value,ensure_ascii=False)));db.commit()
    masks=physical_core_affinities()[:workers];context=multiprocessing.get_context('spawn');queue=context.Queue()
    for mask in masks:queue.put(mask)
    if action=='capture':
        done={int(r[0]) for r in db.execute("SELECT id FROM results WHERE kind='trajectory'")}
        jobs=[(entries[1],seed,str(ppo)) for seed in (20261081,20261083) if seed not in done]
        with ProcessPoolExecutor(max_workers=min(len(masks),2),mp_context=context,initializer=_range_worker_init,initargs=(queue,)) as pool:
            for result in pool.map(_init_screening_capture,jobs):
                for state in result.pop('states'):
                    boundary=pickle.loads(state['context']);key=f"{result['seed']}:{boundary['iteration']}"
                    db.execute('INSERT INTO states VALUES (?,?,?,?,?,?,?)',(key,state['context'],state['pool'],state['c3'],json.dumps(state['c2']),json.dumps(state['c4']),json.dumps(state['online_c4'])))
                save('trajectory',result['seed'],result)
                print(f"S1 native trajectory {result['seed']}: captured, Cmax@60={result['cmax_at60']:.2f}",flush=True)
    elif action=='label':
        done={r[0] for r in db.execute("SELECT id FROM results WHERE kind='screening'")}
        jobs=[(r[0],r[1],r[2],json.loads(r[3]),json.loads(r[4])) for r in db.execute('SELECT id,context,pool,c2,c4 FROM states') if r[0] not in done]
        with ProcessPoolExecutor(max_workers=len(masks),mp_context=context,initializer=_range_worker_init,initargs=(queue,)) as pool:
            for result in pool.map(_init_screening_label,jobs):
                rows=result.pop('rows')
                db.executemany('INSERT INTO labels VALUES (?,?,?)',[(result['state_id'],r['index'],json.dumps(r)) for r in rows])
                save('screening',result['state_id'],result)
                print(json.dumps(result,ensure_ascii=False),flush=True)
    elif action in ('init','run'):
        kind='initialization' if action=='init' else 'search60'
        done={r[0] for r in db.execute('SELECT id FROM results WHERE kind=?',(kind,))}
        if action=='run':
            checks=[json.loads(r[0]) for r in db.execute("SELECT result FROM results WHERE kind='initialization'")]
            if len(checks)!=8 or not all(r['certified'] for r in checks):
                raise ValueError('Complete the eight certified zero-search comparisons before search60')
            i3=next(r for r in checks if r['instance']=='I3' and r['variant']=='IMPROVED')
            if i3['fallback'] or not all(x>0 for x in i3['chosen']['process_loads']):
                raise ValueError('I3 four-robot repair was not effective; search60 not justified')
        jobs=[]
        for i,entry in enumerate(entries):
            seeds=(20261101,) if action=='init' else (20261101,20261102)
            for j,seed in enumerate(seeds):
                variants=('BASELINE','IMPROVED') if (i+j)%2==0 else ('IMPROVED','BASELINE')
                for variant in variants:
                    key=f"{entry['label']}:{seed}:{variant}"
                    if key not in done:
                        job=(entry,variant,str(ppo)) if action=='init' else (entry,seed,variant,str(ppo))
                        jobs.append((key,job))
        worker=_init_screening_init if action=='init' else _init_screening_run
        with ProcessPoolExecutor(max_workers=len(masks),mp_context=context,initializer=_range_worker_init,initargs=(queue,)) as pool:
            futures={pool.submit(worker,job):key for key,job in jobs}
            for future in as_completed(futures):
                result=future.result();save(kind,futures[future],result)
                print(f"{kind} {futures[future]} certified={result.get('certified',result.get('certified_at60'))} Cmax={result.get('cmax_at60',result.get('chosen',{}).get('cmax'))}",flush=True)
    else:
        _init_screening_report(root,db)
    db.close();return 0


def main() -> int:
    import sys
    if "--init-screening" in sys.argv:
        current=argparse.ArgumentParser(description="Current-domain initialization and S1 diagnosis")
        current.add_argument("--init-screening",choices=("capture","label","init","run","report"),required=True)
        current.add_argument("--workers",type=int,default=4)
        args=current.parse_args()
        return _init_screening_main(args.init_screening,args.workers)
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
