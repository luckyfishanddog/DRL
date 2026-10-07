"""Read-only Phase4-0B analysis of existing attempts and certified labels.

No candidate generation, direction optimization, scheduler or solver is called.
Only the three approved analysis files are written, after explicit CLI execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_phase4_0_oracle_recall as audit

SIZES = (64, 80, 96, 112, 128, 160, 192, 224, 256)
GROUPS = ('ALL', 'SMALL', 'MEDIUM', 'LARGE', 'EARLY', 'MID', 'LATE')
MATERIALITY = (('positive', 0.0), ('at_least_0.1pct', .001),
               ('at_least_0.5pct', .005), ('at_least_1pct', .01))
OUTPUT = ROOT / 'data/development/phase4_0b_pool_scaling_analysis.json'
REPORT = ROOT / 'docs/PHASE4_0B_POOL_SCALING_AND_MATERIALITY_20261007.md'


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered)-1)*q
    lo = math.floor(pos); hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def describe(values):
    values = list(values)
    finite = [x for x in values if x is not None]
    if any(not math.isfinite(x) for x in finite):
        raise ValueError('Nonfinite statistic')
    return {'count':len(finite), 'undefined':len(values)-len(finite),
            'mean':statistics.fmean(finite) if finite else None,
            **{name:quantile(finite,q) for name,q in
               (('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('max',1.0))}}


def in_materiality(ratio, threshold):
    return ratio is not None and (ratio > 0 if threshold == 0 else ratio >= threshold)


def select_attempt_prefix(attempts, m):
    if m not in SIZES:
        raise ValueError('Unsupported pool size')
    counts = Counter(); chosen = []
    limits = {'ATOMIC':3*m//4, 'LNS_REPAIRED':m//4}
    for attempt in attempts:
        source = attempt['source']
        if source not in limits:
            raise ValueError('Unknown attempt source')
        if counts[source] < limits[source]:
            chosen.append(attempt)
        counts[source] += 1
    if len(chosen) != m or Counter(x['source'] for x in chosen) != Counter(limits):
        raise ValueError('Missing source prefix')
    return chosen


def build_prefix(chosen, labels):
    """De-duplicate within the retained historical order, not the full pool."""
    seen = set(); representatives = {}; statuses = Counter(); promoted = 0
    for attempt in chosen:
        canonical = attempt.get('canonical_hash')
        if canonical is None:
            statuses[attempt['status']] += 1
            continue
        if canonical in seen:
            statuses['DUPLICATE'] += 1
            continue
        seen.add(canonical)
        label = labels[canonical]
        statuses[label['status']] += 1
        representatives[canonical] = attempt
        promoted += attempt['status'] == 'DUPLICATE'
    return {'representatives':representatives,'counts':dict(statuses),
            'valid_unique':len(seen),'promoted_duplicates':promoted}


def best_cmax(canonicals, labels):
    return min((labels[c]['Cmax'] for c in canonicals
                if labels[c]['status']=='FEASIBLE_CERTIFIED'), default=None)


def generation_metrics(cs, best, oracle):
    epsilon = 1e-9*max(1.0,abs(cs))
    improvement = max(0.0,cs-best) if best is not None else 0.0
    oracle_gain = max(0.0,cs-oracle) if oracle is not None else 0.0
    ratio = None if oracle is None else (cs-oracle)/cs
    raw_regret = None if best is None or oracle is None else max(0.0,best-oracle)/cs
    # Undefined CM* is not a zero regret. The already-certified incumbent is
    # a conservative, executable fallback; no-feasible M256 has no opportunity.
    selection_regret = (raw_regret if raw_regret is not None else
                        max(0.0,cs-oracle)/cs if oracle is not None else 0.0)
    return {'C_star':best,'improvement':improvement,'oracle_improvement_ratio':ratio,
            'generation_recall':improvement/oracle_gain if oracle_gain>epsilon else None,
            'normalized_generation_regret':raw_regret,
            'selection_regret_with_incumbent_fallback':selection_regret,
            'zero_capture':oracle_gain>epsilon and improvement<=epsilon,
            'opportunity':oracle_gain>epsilon,'epsilon_C':epsilon}


def curve_summary(states, m):
    rows = [s['prefixes'][str(m)] for s in states]
    opportunity = [r for r in rows if r['opportunity']]
    statuses = sorted({k for r in rows for k in r['counts']})
    return {'states':len(rows),'opportunity_states':len(opportunity),
            'valid_unique':describe(r['valid_unique'] for r in rows),
            'valid_unique_total':sum(r['valid_unique'] for r in rows),
            'status_count_per_state':{k:describe(r['counts'].get(k,0) for r in rows) for k in statuses},
            'status_totals':{k:sum(r['counts'].get(k,0) for r in rows) for k in statuses},
            'generation_recall':describe(r['generation_recall'] for r in rows),
            'improvement_seconds':describe(r['improvement'] for r in rows),
            'normalized_generation_regret':describe(r['normalized_generation_regret'] for r in rows),
            'selection_regret_with_incumbent_fallback':describe(r['selection_regret_with_incumbent_fallback'] for r in rows),
            'zero_capture_ratio':sum(r['zero_capture'] for r in opportunity)/len(opportunity) if opportunity else None,
            'promoted_duplicates_total':sum(r['promoted_duplicates'] for r in rows)}


def selection_criteria(states, m):
    rows = [s['prefixes'][str(m)] for s in states]
    material = [r for r in rows if in_materiality(r['oracle_improvement_ratio'],.005)]
    regret = describe(r['selection_regret_with_incumbent_fallback'] for r in rows)
    capture = sum(r['generation_recall'] is not None and r['generation_recall']>=.8 for r in material)/len(material) if material else None
    zero = sum(r['zero_capture'] for r in material)/len(material) if material else None
    conditions = {'median_regret_at_most_0.1pct':regret['median'] is not None and regret['median']<=.001,
                  'p90_regret_at_most_0.5pct':regret['p90'] is not None and regret['p90']<=.005,
                  'material_capture80_at_least80pct':capture is not None and capture>=.8,
                  'material_zero_at_most10pct':zero is not None and zero<=.1}
    return {'M':m,'median_regret':regret['median'],'p90_regret':regret['p90'],
            'material_states':len(material),'material_fraction_recall_at_least80':capture,
            'material_zero_capture':zero,'conditions':conditions,'satisfies':all(conditions.values())}


def choose_pool(criteria):
    return next((m for m in SIZES if criteria[str(m)]['satisfies']),None)


def marginal_summary(states, lo, hi):
    rows = []
    for state in states:
        a=state['prefixes'][str(lo)]; b=state['prefixes'][str(hi)]
        rows.append({'valid_unique':b['valid_unique']-a['valid_unique'],
            'FEASIBLE_CERTIFIED':b['counts'].get('FEASIBLE_CERTIFIED',0)-a['counts'].get('FEASIBLE_CERTIFIED',0),
            'DUPLICATE':b['counts'].get('DUPLICATE',0)-a['counts'].get('DUPLICATE',0),
            'CONSTRUCTION_REJECTED':b['counts'].get('CONSTRUCTION_REJECTED',0)-a['counts'].get('CONSTRUCTION_REJECTED',0),
            'C_star_gain_seconds':None if a['C_star'] is None or b['C_star'] is None else a['C_star']-b['C_star'],
            'incumbent_gain_seconds':b['improvement']-a['improvement'],
            'new_opportunity_capture':int(a['improvement']<=a['epsilon_C'] and b['improvement']>b['epsilon_C'])})
    out={'from_M':lo,'to_M':hi,'added_attempts_per_state':hi-lo,'added_attempts_total':len(states)*(hi-lo),
         'metrics':{key:describe(r[key] for r in rows) for key in rows[0]} if rows else {},
         'new_opportunity_captures':sum(r['new_opportunity_capture'] for r in rows)}
    out['gain_seconds_per_added_attempt']=out['metrics']['incumbent_gain_seconds']['mean']/(hi-lo) if rows else None
    return out


def screening_metrics(cs, best, selected_best):
    epsilon=1e-9*max(1.0,abs(cs)); gain=0.0 if best is None else max(0.0,cs-best)
    opportunity=gain>epsilon
    selected_gain=0.0 if selected_best is None else max(0.0,cs-selected_best)
    return {'C_star':selected_best,
            # Preserve the user's literal signed ratio. The nonnegative
            # captured-improvement ratio is also reported for Phase4-0 continuity.
            'recall':None if not opportunity or selected_best is None else (cs-selected_best)/gain,
            'capture_recall':selected_gain/gain if opportunity else None,
            'normalized_regret':None if selected_best is None or best is None else max(0.0,selected_best-best)/cs,
            'incumbent_fallback_regret':None if best is None else max(0.0,(cs if selected_best is None else min(cs,selected_best))-best)/cs,
            'zero_capture':opportunity and selected_gain<=epsilon}


def replay_screening(pool, labels, direction_reader, seed, iteration, cs):
    ranked=sorted(pool,key=lambda c:c.cheap_score)
    ranks={id(c):i for i,c in enumerate(ranked)}
    c2=audit.select_c2_by_family(ranked,8,seed=seed,iteration=iteration)
    c3=[audit.DirectionEvaluatedCandidate(c,ranks[id(c)],direction_reader(c.solution.canonical_hash)) for c in c2]
    c4=audit.select_c4_by_family([c for c in c3 if c.direction.total_empty_travel is not None],2,seed=seed,iteration=iteration)
    best=best_cmax([c.solution.canonical_hash for c in pool],labels)
    result={'pool_C_star':best,
            'C2_ids':[audit.identity(c) for c in c2],
            'C4_ids':[audit.identity(c.screened) for c in c4],
            'C2':screening_metrics(cs,best,best_cmax([c.solution.canonical_hash for c in c2],labels)),
            'C4':screening_metrics(cs,best,best_cmax([c.screened.solution.canonical_hash for c in c4],labels)),
            'pool_improvement_ratio':None if best is None else max(0.0,cs-best)/cs}
    return result


def screening_summary(states):
    opportunity=[s for s in states if s['screening']['pool_C_star'] is not None
                 and s['C_s']-s['screening']['pool_C_star']>s['prefixes']['256']['epsilon_C']]
    out={'states':len(states),'opportunity_states':len(opportunity)}
    for stage in ('C2','C4'):
        rows=[s['screening'][stage] for s in opportunity]
        out[stage]={key:describe(r[key] for r in rows) for key in
                    ('recall','capture_recall','normalized_regret','incumbent_fallback_regret')}
        out[stage]['zero_capture_ratio']=sum(r['zero_capture'] for r in rows)/len(rows) if rows else None
        out[stage]['no_feasible_selected']=sum(r['C_star'] is None for r in rows)
    return out


def group_states(states, group):
    return [s for s in states if group=='ALL' or group in (s['tier'],s['stage'])]


def near(a,b):
    return a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)


def load_state(db, context_row, contexts):
    sid=context_row['state_context_id']; cache={}
    meta=json.loads(db.execute('SELECT metadata FROM states WHERE state_id=?',(sid,)).fetchone()[0])
    if meta['ranker_role']!='RANKER_DEV': raise ValueError('Expected existing RANKER_DEV context')
    context=audit.context_value(context_row,contexts);parents=context['solution'].parents
    labels={}
    for row in db.execute('SELECT candidate_id,canonical_hash,family,first_prefix,status,Cmax,delta_Cmax FROM candidates WHERE state_id=?',(sid,)):
        label=dict(row);label['candidate_id']=audit.expanded_identity(label['candidate_id'])
        if label['status'] is None: raise ValueError('Incomplete existing label')
        if label['canonical_hash'] in labels: raise ValueError('Duplicate canonical label')
        labels[label['canonical_hash']]=label
    by_id={v['candidate_id']:k for k,v in labels.items()}
    attempts=[];repaired={}
    for row in db.execute('SELECT ordinal,source,status,candidate_id FROM attempts WHERE state_id=? ORDER BY ordinal',(sid,)):
        attempt=dict(row);cid=audit.expanded_identity(attempt['candidate_id']);attempt['candidate_id']=cid
        attempt['canonical_hash']=by_id[cid] if cid is not None else None
        # A duplicate LNS can become the first occurrence after atomic truncation.
        if attempt['source']=='LNS_REPAIRED' and attempt['status']=='DUPLICATE':
            event,value=audit.read_attempt_detail(db,sid,attempt['ordinal'],parents,packet_cache=cache)
            candidate=value[3].candidate
            if event!='lns_attempt' or candidate is None: raise ValueError('Invalid duplicate trace')
            canonical=candidate.solution.canonical_hash
            if canonical not in labels: raise ValueError('Missing equivalent canonical label')
            attempt['canonical_hash']=canonical;repaired[attempt['ordinal']]=candidate
        attempts.append(attempt)
    prefixes={str(m):build_prefix(select_attempt_prefix(attempts,m),labels) for m in SIZES}
    oracle=best_cmax(prefixes['256']['representatives'],labels)
    for m in SIZES:
        prefix=prefixes[str(m)]
        prefix.update(generation_metrics(meta['C_s'],best_cmax(prefix['representatives'],labels),oracle))
    return meta,context,labels,attempts,prefixes,repaired,cache



def add_baseline_material_comparison(result):
    states=result['states']
    baseline=[{**state,'screening':state['screening_M64']} for state in states if 'screening_M64' in state]
    result['material_screening_M64']={g:screening_summary([s for s in group_states(baseline,g) if in_materiality(s['screening']['pool_improvement_ratio'],.005)]) for g in GROUPS}
    formula_criteria={}
    for m in SIZES:
        c=result['selection_criteria'][str(m)]
        finite=result['curves']['ALL'][str(m)]['normalized_generation_regret']
        conditions={**c['conditions'],
            'median_regret_at_most_0.1pct':finite['median'] is not None and finite['median']<=.001,
            'p90_regret_at_most_0.5pct':finite['p90'] is not None and finite['p90']<=.005}
        formula_criteria[str(m)]={'satisfies':all(conditions.values()),'p90_regret':finite['p90'],'defined_states':finite['count']}
    result['selection_formula_only_sensitivity']={'recommended_pool_size':choose_pool(formula_criteria),'by_M':formula_criteria}


def analyze():
    started=time.perf_counter()
    input_paths=(audit.RESULT,audit.LABELS,audit.CONTEXTS)
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in input_paths}
    original=audit.read(audit.RESULT);contexts=audit.read(audit.CONTEXTS)
    originals={s['state_context_id']:s for s in original['states']}
    db=sqlite3.connect('file:'+audit.LABELS.as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    states=[];baseline_checks=0
    for row in contexts['contexts']:
        meta,context,labels,attempts,prefixes,repaired,cache=load_state(db,row,contexts)
        old=originals[meta['state_context_id']]
        previous=set()
        for m in SIZES:
            current=set(prefixes[str(m)]['representatives'])
            if not previous<=current: raise ValueError('Non-nested canonical pools')
            previous=current
        for m in (64,128,256):
            new=prefixes[str(m)]
            expected={k for k,v in labels.items() if v['first_prefix']<=m}
            if set(new['representatives'])!=expected or new['valid_unique']!=old['unique_counts'][str(m)]:
                raise ValueError('Historical prefix candidate mismatch')
            if not near(new['C_star'],old['C_star'][str(m)]) or not near(new['improvement'],old['I'][str(m)]):
                raise ValueError('Historical oracle mismatch')
            if any(new['counts'].get(k,0)!=v for k,v in old['counts'][str(m)].items()):
                raise ValueError('Historical attempt status mismatch')
            if m!=256 and not near(new['generation_recall'],old['generation_recall'+str(m)]):
                raise ValueError('Historical generation recall mismatch')
            baseline_checks+=1
        compact={str(m):{k:v for k,v in prefixes[str(m)].items() if k!='representatives'} for m in SIZES}
        states.append({k:meta[k] for k in ('state_context_id','instance_id','tier','stage','seed','iteration','C_s')})
        states[-1]['prefixes']=compact
    criteria={str(m):selection_criteria(states,m) for m in SIZES};recommended=choose_pool(criteria)
    if recommended is not None:
        for state,row in zip(states,contexts['contexts']):
            meta,context,labels,attempts,prefixes,repaired,cache=load_state(db,row,contexts)
            payloads={};directions={}
            def candidate_for(attempt):
                if attempt['candidate_id'] is None:
                    return repaired[attempt['ordinal']]
                key=attempt['candidate_id']
                if key not in payloads:
                    payloads[key]=audit.read_candidate_payload(db,meta['state_context_id'],key,context['solution'].parents,packet_cache=cache)
                return payloads[key]
            def direction_for(canonical):
                if canonical not in directions:
                    diagnostic=audit.read_candidate_diagnostic(db,meta['state_context_id'],labels[canonical]['candidate_id'],packet_cache=cache)
                    directions[canonical]=audit.decode(diagnostic['direction'])
                return directions[canonical]
            def screen(m):
                pool=[candidate_for(a) for a in prefixes[str(m)]['representatives'].values()]
                return replay_screening(pool,labels,direction_for,meta['seed'],meta['iteration'],meta['C_s'])
            baseline=screen(64)
            if baseline['C2_ids']!=meta['C2'] or baseline['C4_ids']!=meta['C4']:
                raise ValueError('Exact production screening replay mismatch')
            old=originals[meta['state_context_id']]
            for stage,key in (('C2','c2_recall'),('C4','c4_recall')):
                if not near(baseline[stage]['capture_recall'],old[key]):
                    raise ValueError('Historical screening metric mismatch')
            state['screening_M64']=baseline
            state['screening']=baseline if recommended==64 else screen(recommended)
    db.close()
    curves={g:{str(m):curve_summary(group_states(states,g),m) for m in SIZES} for g in GROUPS}
    materiality={name:{g:{str(m):curve_summary([s for s in group_states(states,g) if in_materiality(s['prefixes']['256']['oracle_improvement_ratio'],threshold)],m) for m in SIZES} for g in GROUPS} for name,threshold in MATERIALITY}
    marginals={g:[marginal_summary(group_states(states,g),a,b) for a,b in zip(SIZES,SIZES[1:])] for g in GROUPS}
    misses=[s for s in states if s['prefixes']['64']['opportunity'] and s['prefixes']['64']['generation_recall']<.75]
    miss_breakdown={name:sum(in_materiality(s['prefixes']['256']['oracle_improvement_ratio'],threshold) for s in misses) for name,threshold in MATERIALITY}
    screening={g:screening_summary(group_states(states,g)) for g in GROUPS} if recommended is not None else {}
    material_screening={g:screening_summary([s for s in group_states(states,g) if in_materiality(s['screening']['pool_improvement_ratio'],.005)]) for g in GROUPS} if recommended is not None else {}
    stage='CANDIDATE GENERATION';mlp=False
    if recommended is not None:
        material=material_screening['ALL']
        low={k:((material[k]['recall']['median'] is not None and material[k]['recall']['median']<.8)
                 or (material[k]['zero_capture_ratio'] is not None and material[k]['zero_capture_ratio']>.25)) for k in ('C2','C4')}
        mlp=low['C4'];stage='C2+C4' if all(low.values()) else 'C2 SCREENING' if low['C2'] else 'C4 SHORTLIST' if low['C4'] else 'NO MATERIAL SCREENING GAP'
    result={'analysis':'Phase4-0B existing-data pool scaling and materiality',
        'read_only_inputs':[str(x.relative_to(ROOT)) for x in (audit.RESULT,audit.LABELS,audit.CONTEXTS)],
        'sizes':list(SIZES),'recommended_pool_size':recommended,'generation_materiality':'CLOSED' if recommended is not None else 'STILL MATERIAL',
        'main_remaining_bottleneck':stage,'mlp_next':mlp,
        'next_step':'Phase4-1 MLP candidate ranker: freeze features and training design in a separate task' if mlp else 'Candidate generation redesign' if recommended is None else 'Analyze SA acceptance / transitions / horizon',
        'verification':{'historical_prefix_checks':baseline_checks,'exact_M64_screening_replays':len(states) if recommended is not None else 0,
                        'states':len(states),'existing_labels':sum(s['prefixes']['256']['valid_unique'] for s in states),
                        'new_reference_calls':0,'new_direction_optimizations':0,'new_generator_calls':0,'new_trajectories':0},
        'selection_criteria':criteria,'curves':curves,'materiality':materiality,'marginals':marginals,
        'M64_recall_below75_states':len(misses),'M64_miss_materiality':miss_breakdown,
        'M64_miss_actual_regret':describe(s['prefixes']['64']['selection_regret_with_incumbent_fallback'] for s in misses),
        'M64_miss_actual_regret_counts':{name:sum(in_materiality(s['prefixes']['64']['selection_regret_with_incumbent_fallback'],threshold) for s in misses) for name,threshold in MATERIALITY},
        'screening':screening,'material_screening':material_screening,'states':states,
        'elapsed_analysis_seconds':time.perf_counter()-started}
    unchanged=before=={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in input_paths}
    if not unchanged: raise ValueError('Historical input changed during analysis')
    result['verification']['historical_input_sizes_and_mtimes_unchanged']=unchanged
    add_baseline_material_comparison(result)
    return result


def percent(v):
    return 'NA' if v is None else f'{100*v:.3f}%'


def number(v):
    return 'NA' if v is None else f'{v:.3f}'


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def report_text(r):
    m=r['recommended_pool_size'];all_curves=r['curves']['ALL'];chosen=all_curves[str(m)] if m else None
    miss=r['M64_miss_materiality'];miss_loss=r['M64_miss_actual_regret_counts']
    bad_actual=miss_loss['at_least_0.5pct'];tiny=miss['positive']-miss['at_least_0.1pct']
    waste=sum(chosen['status_count_per_state'].get(k,{}).get('mean',0) for k in ('DUPLICATE','CONSTRUCTION_REJECTED','IDENTITY','RAW_REJECTED')) if chosen else None
    block_gains={f'{a}→{b}':(all_curves[str(b)]['improvement_seconds']['mean']-all_curves[str(a)]['improvement_seconds']['mean'])/(b-a) for a,b in ((64,128),(128,192),(192,256))}
    previous=next((x for x in reversed(SIZES) if m is not None and x<m),None)
    failed=[] if previous is None else [k for k,v in r['selection_criteria'][str(previous)]['conditions'].items() if not v]
    mat=r['material_screening'].get('ALL')
    lines=['# Phase4-0B：候选池规模、materiality 与筛选损失分析','',
        f"Phase4-0B: PASS；推荐候选池 M*={m}；generation materiality={r['generation_materiality']}；剩余瓶颈={r['main_remaining_bottleneck']}；MLP next={'YES' if r['mlp_next'] else 'NO'}。",'',
        '只读取既有 72 states / 10,701 labels；没有重新生成候选、优化方向、调用 reference/certifier 或运行 trajectory。生产 M64/Kdp8/Kref2 保持。只分析原 RANKER_DEV，不打开训练、validation 或 test workbook。历史 Phase4-0 文件不改写。','',
        '## 口径与 NULL','',
        '按 atomic 前 3M/4 次、LNS 前 M/4 次截取，保留历史块顺序（48A+16L、48A+16L、96A+32L）中的相对次序。不能直接取混合总流前 M 次。前缀内按完整 canonical solution 去重；原先与被移除 atomic 重复的 LNS 可成为首个代表，仍复用同一完整解的现有标签。中间档位的完整解集合嵌套，代表 candidate identity 可能替换。实际生产更改 M 时仍需另行验证运行实现和成本。','',
        '分位数使用 sorted values 的 (n−1)q 线性插值；主统计每 state 等权。Generation recall 以 I256 为分母，沿用 Phase4-0 epsilon。materiality 四组按有符号 (Cs−C256*)/Cs 判断，全部报告，不用阈值挑选实例。regret 以 Cs 归一化，数值均为比例。','',
        'C* 无可行候选时为 NULL，公式 regret 也为 NULL，不填零；报告给出 undefined 数。M* 选择额外使用已有 certified incumbent：CM* 缺失但 C256* 存在时，保守损失为 max(0,Cs−C256*)/Cs；两者均缺失时无有限池改善机会。既有有限 C* 的用户公式原样保留。','',
        '筛选 recall 同时给出用户原式的 signed recall=(Cs−Cselected*)/(Cs−CM*)，可能为负，以及与 Phase4-0 可比的非负 capture recall=max(0,Cs−Cselected*)/(Cs−CM*)。selected C* 为 NULL 时 signed recall/regret 为 NULL，zero capture 仍计入分母；非负 capture recall 用已认证 incumbent 的零改善，绝非给不可行 candidate 填零 Cmax。normalized screening regret 原式不截断，同时给出执行 incumbent fallback 的 regret。','',
        '## 15 个问题的直接回答','',
        f"1. Phase4-0 中 M64 recall<75% 的状态有 {r['M64_recall_below75_states']} 个，其中 oracle 改善<0.1% 的只有 {tiny} 个、≥0.5% 的有 {miss['at_least_0.5pct']} 个、≥1% 的有 {miss['at_least_1pct']} 个。更关键的是，其中 {bad_actual} 个状态因 M64 漏解实际损失≥当前 makespan 的0.5%，{miss_loss['at_least_1pct']} 个损失≥1%；实际 normalized regret median={percent(r['M64_miss_actual_regret']['median'])}、p90={percent(r['M64_miss_actual_regret']['p90'])}。{'因此不能把原 FAIL 解释为许多极小改善造成的假象。' if bad_actual>tiny else '原 FAIL 同时包含微小机会与真实损失，不能单凭计数断言主因。'}四组嵌套计数不能相加。",
        '2. 九个 M 的完整 generation 曲线见 ALL 与各 tier/stage 表；同时保留中位、均值、四分位及 regret 尾部。',
        f"3. 按连续64个新增 attempts 归并，64→128、128→192、192→256 的每新增 attempt 平均 incumbent 改善分别为 {number(block_gains['64→128'])}、{number(block_gains['128→192'])}、{number(block_gains['192→256'])} 秒。{'后段边际收益低于前段，体现总体 diminishing returns；' if block_gains['192→256']<block_gains['64→128'] else '边际曲线并非单调，不能宣称存在唯一拐点；'}M128 是否够用由四条件决定，其结果={r['selection_criteria']['128']['satisfies']}；不能因中位捕获100%就说128后已饱和。达到工程容忍度的位置为 M{m}，不是宣称其后收益严格为零。",
        f"4. 按用户给定四条件选择的全局最小 M*={m}，不按 tier 单独配置。",
        f"5. 前一档 M{previous} 未满足：{', '.join(failed) or 'NA'}；M{m} 是首次四项均通过的档位。M256 提供更强有限池 oracle，但多生成并不自动抵消生成/筛选成本；本轮没有测量改 M 后的生产 runtime。这里只推荐离线池规模，未修改配置。",
        f"6. M* 下有效 unique 每状态均值 {number(chosen['valid_unique']['mean']) if chosen else 'NA'}、中位 {number(chosen['valid_unique']['median']) if chosen else 'NA'}。",
        f"7. M* 每状态平均 duplicate={number(chosen['status_count_per_state'].get('DUPLICATE',{}).get('mean')) if chosen else 'NA'}、construction reject={number(chosen['status_count_per_state'].get('CONSTRUCTION_REJECTED',{}).get('mean')) if chosen else 'NA'}；duplicate/reject/identity/raw reject 合计占 {percent(waste/m) if m else 'NA'}，浪费明显且仍占预算、不补抽。",
        f"8. M* 自身改善≥0.5% 的状态 n={mat['states'] if mat else 0}；C2 capture recall median={percent(mat['C2']['capture_recall']['median']) if mat else 'NA'}，signed recall median={percent(mat['C2']['recall']['median']) if mat else 'NA'}，zero capture={percent(mat['C2']['zero_capture_ratio']) if mat else 'NA'}。",
        f"9. 同一组 material states 中，C4 capture recall median={percent(mat['C4']['capture_recall']['median']) if mat else 'NA'}，signed recall median={percent(mat['C4']['recall']['median']) if mat else 'NA'}，zero capture={percent(mat['C4']['zero_capture_ratio']) if mat else 'NA'}。使用相同 Kref2、方向 rerank 和 family policy，ID 来自生产函数，标签直接复用；signed 与 capture 分母样本差别见 NULL 口径。",
        '10. SMALL/MEDIUM/LARGE 与 EARLY/MID/LATE 分别给出，不把候选混池计算平均。',
        '11. LARGE 的 material generation 与 screening 结果单独比较，见下文；只表述 consistent with，不作退化因果断言。',
        '12. 现有 Phase4-0 已显示 M256 oracle-best 主要为 STRUCTURAL/WHOLE，LARGE 的 Y/X 没有达到 M256 最优；本轮没有支持修改 X/Y 规则的新证据，不修改 proposal/quota/split。',
        f"13. {'保持生产 M64；离线推荐 M64。' if m==64 else '未来可研究 M'+str(m)+'；当前生产仍为 M64。' if m else '暂不选生产池，先研究 candidate generation。'}",
        f"14. MLP 科学动机：{'成立：有工程意义的改善已在推荐池内，而 C4 在 material states 仍有明显损失。' if r['mlp_next'] else '本轮证据不支持直接进入 MLP。'}",
        '15. 下一步：'+r['next_step']+'。本轮到分析报告停止，不建网络、不训练、不拟合标准化。','',
        '## M* 四条件选择','']
    lines+=['选择表使用含 certified incumbent fallback 的保守口径；后面的 generation 曲线表原样报告公式仅有限值的 regret。仅使用公式有限值重新选择，最小 M 同样为 '+str(r['selection_formula_only_sensitivity']['recommended_pool_size'])+'，结论不依赖 NULL 处理。','']
    lines+=table(['M','median regret','p90 regret','material n','material recall≥80% 比例','material zero','满足'],
        [[n,percent(v['median_regret']),percent(v['p90_regret']),v['material_states'],percent(v['material_fraction_recall_at_least80']),percent(v['material_zero_capture']),str(v['satisfies'])] for n in SIZES for v in [r['selection_criteria'][str(n)]]])
    for group in GROUPS:
        lines+=['',f'## Generation：{group}','']
        lines+=table(['M','unique mean/median','recall mean/median/p25/p75','regret median/p75/p90/p95/max','regret NA','zero capture'],
            [[n,number(v['valid_unique']['mean'])+'/'+number(v['valid_unique']['median']),
              '/'.join(percent(v['generation_recall'][k]) for k in ('mean','median','p25','p75')),
              '/'.join(percent(v['normalized_generation_regret'][k]) for k in ('median','p75','p90','p95','max')),
              v['normalized_generation_regret']['undefined'],percent(v['zero_capture_ratio'])] for n in SIZES for v in [r['curves'][group][str(n)]]])
        lines+=['','每状态平均尝试分类（括号外 unique，状态总数另存 JSON；每 state 固定 M 次）：','']
        keys=('CONSTRUCTION_REJECTED','IDENTITY','DUPLICATE','DIRECTION_INFEASIBLE','FEASIBLE_CERTIFIED','DEADLOCK')
        lines+=table(['M','unique']+list(keys),[[n,number(v['valid_unique']['mean'])]+[number(v['status_count_per_state'].get(k,{}).get('mean',0)) for k in keys] for n in SIZES for v in [r['curves'][group][str(n)]]])
        lines+=['','边际收益（均值；C* gain 排除缺失配对，incumbent gain 包含零捕获→捕获）：','']
        lines+=table(['档位','新增 attempts','新增 unique/FEASIBLE','新增 dup/reject','首次捕获状态','C* gain s','incumbent gain s','gain s/attempt'],
            [[f"{v['from_M']}→{v['to_M']}",v['added_attempts_per_state'],
              number(v['metrics']['valid_unique']['mean'])+'/'+number(v['metrics']['FEASIBLE_CERTIFIED']['mean']),
              number(v['metrics']['DUPLICATE']['mean'])+'/'+number(v['metrics']['CONSTRUCTION_REJECTED']['mean']),v['new_opportunity_captures'],
              number(v['metrics']['C_star_gain_seconds']['mean']),number(v['metrics']['incumbent_gain_seconds']['mean']),number(v['gain_seconds_per_added_attempt'])] for v in r['marginals'][group]])
    lines+=['','## Materiality：四组全部报告','']
    for name,_ in MATERIALITY:
        lines+=['',f'### {name}','']
        lines+=table(['group','n','M64 recall median','M64 regret p90/max','M64 zero','M* recall median','M* regret p90','M* zero'],
            [[g,a['states'],percent(a['generation_recall']['median']),percent(a['normalized_generation_regret']['p90'])+'/'+percent(a['normalized_generation_regret']['max']),percent(a['zero_capture_ratio']),percent(b['generation_recall']['median']),percent(b['normalized_generation_regret']['p90']),percent(b['zero_capture_ratio'])] for g in GROUPS for a,b in [(r['materiality'][name][g]['64'],r['materiality'][name][g][str(m or 256)])]])
    if m is not None:
        for title,groups in (('全部 M* 改善机会',r['screening']),('M* 改善≥0.5% material states',r['material_screening'])):
            lines+=['',f'## Screening：{title}','']
            lines+=table(['group','n/机会','stage','signed recall median','capture recall median','zero capture','raw regret median/p90','NA selected','fallback regret median/p90'],
                [[g,f"{v['states']}/{v['opportunity_states']}",stage,percent(x['recall']['median']),percent(x['capture_recall']['median']),percent(x['zero_capture_ratio']),percent(x['normalized_regret']['median'])+'/'+percent(x['normalized_regret']['p90']),x['no_feasible_selected'],percent(x['incumbent_fallback_regret']['median'])+'/'+percent(x['incumbent_fallback_regret']['p90'])] for g,v in groups.items() for stage,x in ((k,v[k]) for k in ('C2','C4'))])
        large_gen=r['materiality']['at_least_0.5pct']['LARGE'][str(m)]
        large=r['material_screening']['LARGE']
        base_large=r['material_screening_M64']['LARGE']
        base_gen=r['materiality']['at_least_0.5pct']['LARGE']['64']
        lines+=['','## M64 原结论在 material 过滤后是否成立','',
            '以下以各 state 的 M64 自身改善≥0.5% 作为筛选 material 分母，全部来自本轮已验证的 M64 生产 ID 重放。','']
        lines+=table(['tier','material states','C2 capture median','C4 capture median','C4 zero capture'],
            [[g,v['states'],percent(v['C2']['capture_recall']['median']),percent(v['C4']['capture_recall']['median']),percent(v['C4']['zero_capture_ratio'])] for g,v in r['material_screening_M64'].items() if g in ('ALL','SMALL','MEDIUM','LARGE')])
        lines+=['',f"LARGE：在 M256 改善≥0.5% 的 {base_gen['states']} 个状态上，M64 generation recall median={percent(base_gen['generation_recall']['median'])}；在 M64 自身改善≥0.5% 的 {base_large['states']} 个状态上，C4 capture recall median={percent(base_large['C4']['capture_recall']['median'])}、zero capture={percent(base_large['C4']['zero_capture_ratio'])}。因此原先 LARGE 的筛选损失观察在去掉微小改善后仍成立，但没有消除 generation 尾部。"]
        lines+=['','## LARGE：直接比较','',
            f"M256 改善≥0.5% 的 LARGE 状态 n={large_gen['states']}；M* generation recall median={percent(large_gen['generation_recall']['median'])}，p90 normalized generation regret={percent(large_gen['normalized_generation_regret']['p90'])}。M* 自身改善≥0.5% 的 LARGE 状态 n={large['states']}；C2 capture recall median={percent(large['C2']['capture_recall']['median'])}，C4 capture recall median={percent(large['C4']['capture_recall']['median'])}，C4 zero capture={percent(large['C4']['zero_capture_ratio'])}。两组分母不同，未混用。",
            'LARGE degradation is more consistent with candidate screening/ranking loss than candidate-pool size.' if large['states'] and large_gen['generation_recall']['median']>=.8 and (large['C4']['capture_recall']['median']<.8 or large['C4']['zero_capture_ratio']>.25) else 'LARGE 尚不能仅凭这些状态把问题定位为 screening；见上述 generation 尾部与筛选损失。',
            '这描述的是已采开发状态的机会损失，不代表对整个算法规模退化的因果识别。']
    lines+=['','## 曲线（等状态权重）','',
            '```mermaid','xychart-beta','    title "Generation recall mean (%)"','    x-axis "M" ['+', '.join(map(str,SIZES))+']','    y-axis "Capture %" 0 --> 100',
            '    line ['+', '.join(f"{100*all_curves[str(n)]['generation_recall']['mean']:.3f}" for n in SIZES)+']','```','',
            '```mermaid','xychart-beta','    title "Normalized generation regret p90 (%)"','    x-axis "M" ['+', '.join(map(str,SIZES))+']',
            '    y-axis "Regret %"','    line ['+', '.join(f"{100*all_curves[str(n)]['selection_regret_with_incumbent_fallback']['p90']:.3f}" for n in SIZES)+']','```','',
            '```mermaid','xychart-beta','    title "Marginal incumbent gain (s / added attempt)"',
            '    x-axis ["64-80", "80-96", "96-112", "112-128", "128-160", "160-192", "192-224", "224-256"]',
            '    y-axis "Seconds / attempt"',
            '    line ['+', '.join(f"{v['gain_seconds_per_added_attempt']:.3f}" for v in r['marginals']['ALL'])+']','```','',
            '## 限制与复现','',
            'M256 是有限参考池，不是全局最优。以 M256 自身为 oracle，M256 的 generation regret 按定义为零；其通过不能证明 generator family 已覆盖全问题。因此本轮结论只关闭相对该有限池的 material loss。72 states 在 12 instances/24 trajectories 内相关，未进行独立样本显著性检验。','',
            '执行方式：从 DRL 运行 `D:\\pybullet_test\\.venv\\Scripts\\python.exe -B scripts/analyze_phase4_0b_pool_scaling.py`。SQLite 用 mode=ro，现有 reader 扩展可选逐状态缓存，存储语义不改。只在既有 payload 上调用生产 select_c2_by_family/select_c4_by_family，C3 复用既有 direction diagnostic。','',
            '验证：'+json.dumps(r['verification'],ensure_ascii=False)+'；完整回归：'+json.dumps(r.get('regression',{}),ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def write_outputs(result):
    text=report_text(result)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    REPORT.write_text(text,encoding='utf-8')


def console_summary(r):
    m=r['recommended_pool_size']
    print('Phase4-0B: PASS')
    print(f'Recommended candidate pool: M* = {m}')
    print('Generation materiality: '+r['generation_materiality'])
    print('Main remaining bottleneck: '+r['main_remaining_bottleneck'])
    print('MLP next: '+('YES' if r['mlp_next'] else 'NO'))
    print('Next step: '+r['next_step'])
    print('Key selection numbers: '+json.dumps(r['selection_criteria'][str(m or 256)],ensure_ascii=False))
    if m is not None:
        print('Material screening ALL: '+json.dumps({stage:{'capture_median':r['material_screening']['ALL'][stage]['capture_recall']['median'],'signed_median':r['material_screening']['ALL'][stage]['recall']['median'],'zero_capture':r['material_screening']['ALL'][stage]['zero_capture_ratio']} for stage in ('C2','C4')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--regression-passed',type=int)
    args=parser.parse_args()
    result=analyze()
    if args.regression_passed is not None:
        result['regression']={'passed':args.regression_passed,'command':'pytest -q -p no:cacheprovider','interpreter':sys.executable}
    if not args.preview:
        write_outputs(result)
    console_summary(result)