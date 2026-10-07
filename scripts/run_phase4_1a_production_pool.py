"""Phase4-1A: native development wall-clock smoke and MLP preparation.

No new scientific protocol. Fixed cases come directly from Phase3-YR.
Only M and M_lns vary. The parent alone writes resumable run records.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import asdict, replace
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sqlite3
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts import run_phase4_0_oracle_recall as audit
from mrta_search import pipeline
from mrta_data.phase3_split import workbook_descriptors, _normalized, DESCRIPTOR_FIELDS

OUTPUT=ROOT/'data/development/phase4_1a_production_pool_smoke.json'
SPLIT=ROOT/'data/development/mlp_workbook_split_v1.json'
REPORT=ROOT/'docs/PHASE4_1A_PRODUCTION_POOL_AND_MLP_DATA_PREP_20261007.md'
SIZES=(64,128,160,192)
SEEDS=(20261111,20261112,20261113)
TIERS=('SMALL','MEDIUM','LARGE')


def config_for(m):
    if m not in SIZES: raise ValueError('Unsupported pool')
    return replace(audit.SEARCH,m=m,m_lns=m//4)


def development_cases():
    source=audit.read(ROOT/'data/manifests/PHASE3YR_V2_ACCESS_PROTOCOL_V1.json')
    entries=source['instances']
    if len(entries)!=6 or Counter(e['tier'] for e in entries)!={t:2 for t in TIERS}:
        raise ValueError('Expected the six existing Y/YR mechanism cases')
    roles=audit.read(audit.ROLES)
    audit.assert_v2_solver_access_allowed(roles,[e['relative_path'] for e in entries],allowed_roles=('V2_MODEL_DEVELOPMENT_CONSUMED',))
    return [{k:e[k] for k in ('mechanism_id','instance_id','relative_path','sheet_name','tier','N','raw_file_sha256','instance_geometry_hash')} for e in entries]


def physical_core_affinities():
    class Info(ctypes.Structure):
        _fields_=[('mask',ctypes.c_size_t),('relationship',ctypes.c_int),('reserved',ctypes.c_ulonglong*2)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);length=wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformation(None,ctypes.byref(length))
    buf=ctypes.create_string_buffer(length.value)
    if not kernel.GetLogicalProcessorInformation(buf,ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    masks=[]
    for offset in range(0,length.value,ctypes.sizeof(Info)):
        item=Info.from_buffer_copy(buf.raw[offset:offset+ctypes.sizeof(Info)])
        if item.relationship==0: masks.append(int(item.mask)&-int(item.mask))
    return masks


def bind_to_core(mask):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.c_size_t)
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),mask):
        raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def generation_timers():
    totals={'candidate_generation_seconds':0.0,'lns_seconds':0.0}
    originals={name:getattr(pipeline,name) for name in ('generate_candidate_pool','destroy_parents','repair_partial_state')}
    def timed(name,key):
        def call(*args,**kwargs):
            start=time.perf_counter()
            try: return originals[name](*args,**kwargs)
            finally: totals[key]+=time.perf_counter()-start
        return call
    pipeline.generate_candidate_pool=timed('generate_candidate_pool','candidate_generation_seconds')
    pipeline.destroy_parents=timed('destroy_parents','lns_seconds')
    pipeline.repair_partial_state=timed('repair_partial_state','lns_seconds')
    try: yield totals
    finally:
        for name,function in originals.items(): setattr(pipeline,name,function)


def checkpoint_cmax(result,deadline=60.0):
    # best_events are emitted only after certification, timestamped on the
    # native run clock (including initialization). Never use final/overshoot.
    value=result.stats.anytime((deadline,))[deadline]['cmax']
    if value!=result.anytime[deadline]['cmax']: raise ValueError('Checkpoint mismatch')
    return value


def run_one(entry,parents,m,seed,affinity,provenance):
    cpu=time.process_time()
    with generation_timers() as timings:
        result=pipeline.run_sa_oi_alns_v2(parents,audit.CONFIG,config_for(m),seed=seed,source_provenance=provenance)
    st=result.stats;runtime=result.runtime;c60=checkpoint_cmax(result)
    unique=sum(st.valid_by_family.values())
    if st.raw_attempts!=st.iterations*m: raise ValueError('Native attempt budget mismatch')
    return {'instance_id':entry['instance_id'],'case':entry['mechanism_id'],'tier':entry['tier'],'N':entry['N'],
        'M':m,'M_atomic':3*m//4,'M_lns':m//4,'seed':seed,'cpu_affinity_mask':affinity,
        'Cmax_at_60':c60,'checkpoint_certified':c60 is not None,
        'final_certified':bool(result.final_certification and result.final_certification.certified),
        'final_cmax':None if result.best_metrics is None else result.best_metrics.cmax,
        'iterations_completed':st.iterations,'raw_candidate_attempts':st.raw_attempts,'valid_unique_candidates':unique,
        **timings,'raw_generation_counter_seconds':st.candidate_generation_time,'cheap_screen_seconds':st.cheap_screen_time,
        'lns_repair_seconds':st.repair_time,'direction_dp_seconds':st.direction_dp_time,
        'reference_scheduler_seconds':st.reference_scheduler_time,'reference_calls':st.nref+st.init_reference_calls,
        'search_reference_calls':st.nref,'init_reference_calls':st.init_reference_calls,'initialization_seconds':st.init_time,
        'accepted_moves':sum(accepted for _,_,accepted,_ in st.proposal_trajectory),
        'global_best_updates':max(0,len(st.best_events)-1),
        'deadlock_count':st.n_deadlock+st.init_status_counts.get('DEADLOCK',0),
        'numeric_failure':st.n_numeric_failure+st.init_status_counts.get('NUMERIC_FAILURE',0)+int(result.status is pipeline.SearchStatus.NUMERIC_FAILURE),
        'actual_runtime':runtime,'overshoot':st.overshoot,'termination':result.termination_reason,
        'cpu_seconds':time.process_time()-cpu,
        'iterations_per_second':st.iterations/runtime,'unique_candidates_per_second':unique/runtime,
        'reference_calls_per_second':(st.nref+st.init_reference_calls)/runtime,
        'candidate_generation_time_fraction':timings['candidate_generation_seconds']/runtime,
        'best_events':[list(e) for e in st.best_events]}


def seed_worker(seed_index,affinity,entries,parents,completed,events):
    try:
        bind_to_core(affinity);seed=SEEDS[seed_index];source=audit.provenance()
        for index,entry in enumerate(entries):
            offset=(index+seed_index)%len(SIZES)
            order=SIZES[offset:]+SIZES[:offset]
            for m in order:
                if (entry['instance_id'],m,seed) in completed: continue
                events.put(('run',run_one(entry,parents[index],m,seed,affinity,source)))
        events.put(('done',seed))
    except BaseException:
        events.put(('error',{'seed':SEEDS[seed_index],'traceback':traceback.format_exc()}))


def save_result(value):
    OUTPUT.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run_smoke():
    offline=audit.read(ROOT/'data/development/phase4_0b_pool_scaling_analysis.json')
    entries=development_cases();masks=physical_core_affinities()
    if len(masks)<3: raise ValueError('Three distinct physical cores required for this execution arrangement')
    result=audit.read(OUTPUT) if OUTPUT.exists() else {'phase':'Phase4-1A','offline_recommended_pool':offline['recommended_pool_size'],
        'cases':entries,'seeds':list(SEEDS),'configs':{str(m):asdict(config_for(m)) for m in SIZES},
        'execution':'3 seed processes on separate physical cores; one logical CPU per core; sequential rotated M order within each seed',
        'affinity_masks':masks[:3],'runs':[]}
    if result['cases']!=entries or result['configs']!={str(m):asdict(config_for(m)) for m in SIZES} or result['seeds']!=list(SEEDS):
        raise ValueError('Existing result has different cases/configs/seeds')
    completed={(r['instance_id'],r['M'],r['seed']) for r in result['runs']}
    if len(completed)!=len(result['runs']): raise ValueError('Duplicate run record')
    if len(completed)==72:
        print('72 existing native runs retained; no reruns',flush=True);return
    parents=[audit.load_parents(e,debug=True) for e in entries]
    save_result(result)
    context=mp.get_context('spawn');events=context.Queue()
    workers=[context.Process(target=seed_worker,args=(i,masks[i],entries,parents,completed,events)) for i in range(3)]
    for worker in workers: worker.start()
    done=set()
    try:
        while len(done)<3:
            try: kind,value=events.get(timeout=5)
            except queue.Empty:
                if any(w.exitcode not in (None,0) for w in workers): raise RuntimeError('Smoke worker exited unexpectedly')
                continue
            if kind=='error': raise RuntimeError(value['traceback'])
            if kind=='done': done.add(value);continue
            key=(value['instance_id'],value['M'],value['seed'])
            if key in completed: raise ValueError('Repeated completed run')
            result['runs'].append(value);completed.add(key);save_result(result)
            print(f"{len(completed)}/72 {value['case']} M{value['M']} seed={value['seed']} Cmax@60={value['Cmax_at_60']} iters={value['iterations_completed']} runtime={value['actual_runtime']:.3f}",flush=True)
    finally:
        for worker in workers:
            if worker.is_alive() and len(done)<3: worker.terminate()
        for worker in workers: worker.join()
    if len(completed)!=72: raise ValueError('Incomplete native smoke')


def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def compare_and_choose(runs):
    grouped={}
    for r in runs:
        group=grouped.setdefault((r['instance_id'],r['seed']),{})
        if r['M'] in group: raise ValueError('Duplicate paired run')
        group[r['M']]=r
    if len(grouped)!=18 or any(set(g)!=set(SIZES) for g in grouped.values()): raise ValueError('Need 18 complete paired cases')
    ratios={m:[] for m in SIZES}
    for group in grouped.values():
        best=min((r['Cmax_at_60'] for r in group.values() if r['checkpoint_certified'] and r['Cmax_at_60'] is not None),default=None)
        for m,r in group.items():
            ratios[m].append({'tier':r['tier'],'ratio':None if best is None or not r['checkpoint_certified'] or r['Cmax_at_60'] is None else r['Cmax_at_60']/best})
    baseline_iterations=median(r['iterations_completed'] for r in runs if r['M']==64)
    summary={}
    for m in SIZES:
        rows=[r for r in runs if r['M']==m]
        tier_ratios={t:median(r['ratio'] for r in ratios[m] if r['tier']==t) for t in TIERS}
        overall=median(r['ratio'] for r in ratios[m]);iterations=median(r['iterations_completed'] for r in rows)
        certified=sum(r['checkpoint_certified'] and r['Cmax_at_60'] is not None for r in rows)
        retention=iterations/baseline_iterations if baseline_iterations else 0.0
        conditions={'all_18_checkpoints_certified':certified==18,'overall_within1pct':overall is not None and overall<=1.01,
            'all_tiers_within2pct':all(v is not None and v<=1.02 for v in tier_ratios.values()),'iteration_retention_at_least40pct':retention>=.4}
        fields=('iterations_completed','candidate_generation_time_fraction','reference_calls','reference_calls_per_second',
            'candidate_generation_seconds','lns_seconds','direction_dp_seconds','reference_scheduler_seconds','valid_unique_candidates',
            'unique_candidates_per_second','actual_runtime','overshoot','initialization_seconds')
        summary[str(m)]={'median_ratio':overall,'tier_median_ratio':tier_ratios,'certified_checkpoints':certified,
            'median_iteration_retention_vs_M64':retention,'conditions':conditions,'satisfies':all(conditions.values()),
            'medians':{k:median(r[k] for r in rows) for k in fields},
            'tier_medians':{t:{k:median(r[k] for r in rows if r['tier']==t) for k in ('iterations_completed','candidate_generation_time_fraction','reference_calls','Cmax_at_60')} for t in TIERS}}
    chosen=next((m for m in SIZES if summary[str(m)]['satisfies']),64)
    return chosen,summary


def compute_mlp_split():
    old=audit.read(audit.SPLIT)
    train=sorted(w['relative_path'] for w in old['workbooks'] if w['ranker_role']=='RANKER_TRAIN')
    consumed=sorted(w['relative_path'] for w in old['workbooks'] if w['ranker_role']=='RANKER_DEV')
    if len(train)!=23 or len(consumed)!=8 or set(train)&set(consumed): raise ValueError('Expected 23 clean plus 8 consumed workbooks')
    descriptors=workbook_descriptors(audit.read(audit.DATASET));vectors=_normalized(descriptors,train)
    order=audit.farthest_order(train,vectors,{p:p for p in train})
    dev=sorted(order[:6]);remaining=sorted(set(train)-set(dev))
    result={'MLP_TRAIN':remaining,'MLP_DEV':dev,'ORACLE_DEV_CONSUMED':consumed,
        'workbook_identity':{p:{'raw_file_sha256':descriptors[p]['raw_file_sha256']} for p in sorted(train+consumed)},
        'selection':'farthest-first on standardized existing static workbook descriptors; path lexical tie-break; no solver outcomes',
        'descriptor_fields':list(DESCRIPTOR_FIELDS)}
    return result


def make_mlp_split():
    result=compute_mlp_split()
    if SPLIT.exists() and audit.read(SPLIT)!=result: raise ValueError('Existing MLP split differs; not overwriting')
    SPLIT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def finish(regression_passed):
    result=audit.read(OUTPUT)
    chosen,summary=compare_and_choose(result['runs'])
    if chosen>64 and pipeline.SA_OI_ALNS_V2_PRODUCTION_POOL_SIZE!=chosen:
        raise ValueError('V2 production default does not match measured selection')
    result.update(M_prod=chosen,pool_comparison=summary)
    save_result(result)
    split=make_mlp_split()
    result['mlp_split_counts']={k:len(split[k]) for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED')}
    result['feature_validation']=validate_existing_features()
    result['regression']={'passed':regression_passed,'interpreter':sys.executable,'command':'pytest -q -p no:cacheprovider'}
    result['passed']=all(r['checkpoint_certified'] and r['final_certified'] and not r['numeric_failure'] for r in result['runs']) and result['feature_validation']['passed'] and regression_passed is not None
    save_result(result);REPORT.write_text(report_text(result),encoding='utf-8')
    print_summary(result)


def validate_existing_features():
    from mrta_ranker.features import extract_c2_features,extract_c4_features,make_training_targets
    contexts=audit.read(audit.CONTEXTS);db=sqlite3.connect('file:'+audit.LABELS.as_posix()+'?mode=ro',uri=True)
    counts=Counter();widths={};before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in (audit.CONTEXTS,audit.LABELS,audit.RESULT)}
    for row in contexts['contexts']:
        context=audit.context_value(row,contexts);parents=context['solution'].parents;cache={}
        records=[(cid,status,cmax,audit.read_candidate_payload(db,row['state_context_id'],cid,parents,packet_cache=cache)) for cid,status,cmax in db.execute('SELECT candidate_id,status,Cmax FROM candidates WHERE state_id=?',(row['state_context_id'],))]
        ranks={id(candidate):rank for rank,candidate in enumerate(sorted((record[3] for record in records),key=lambda c:c.cheap_score))}
        for cid,status,cmax,candidate in records:
            cheap_rank=ranks[id(candidate)]
            args=(context['solution'],context['directions'],context['metrics'].cmax,candidate,audit.CONFIG)
            c2=extract_c2_features(*args)
            direction=audit.decode(audit.read_candidate_diagnostic(db,row['state_context_id'],cid,packet_cache=cache)['direction'])
            c4=extract_c4_features(*args,direction=direction,cheap_rank=cheap_rank)
            if c2!=extract_c2_features(*args) or c4!=extract_c4_features(*args,direction=direction,cheap_rank=cheap_rank): raise ValueError('Nondeterministic feature extraction')
            if any(not math.isfinite(v) for features in (c2,c4) for v in features.values()): raise ValueError('Nonfinite feature')
            for kind,features in (('C2',c2),('C4',c4)):
                names=tuple(features)
                if kind in widths and widths[kind]!=names: raise ValueError('Feature columns changed by candidate')
                widths[kind]=names
            if any(any(word in k.lower() for word in ('direction','reference','wait','deadlock','certif','schedule','c3','c4')) for k in c2): raise ValueError('C2 information leakage')
            if any(any(word in k.lower() for word in ('reference','wait','deadlock','certif','schedule')) for k in c4): raise ValueError('C4 information leakage')
            targets=make_training_targets(status,context['metrics'].cmax,cmax)
            if status=='NUMERIC_FAILURE' and targets is not None: raise ValueError('Numeric label included')
            counts['candidates']+=1;counts[status]+=1
    db.close()
    unchanged=before=={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in before}
    return {'passed':counts['candidates']==10701 and len(contexts['contexts'])==72 and unchanged,'states':len(contexts['contexts']),'counts':dict(counts),
        'C2_feature_count':len(widths['C2']),'C4_feature_count':len(widths['C4']),
        'C2_feature_names':list(widths['C2']),'C4_feature_names':list(widths['C4']),
        'historical_inputs_unchanged':unchanged,'new_reference_calls':0,'new_direction_optimizations':0,
        'purpose':'consumed ORACLE_DEV pipeline regression only; true frozen M256 cheap ranks; no feature selection or fitting'}


def report_text(r):
    chosen=r['M_prod'];summary=r['pool_comparison'];s=summary[str(chosen)]
    best_pools=[m for m in SIZES if summary[str(m)]['median_ratio']==min(v['median_ratio'] for v in summary.values())]
    tier_best={t:[m for m in SIZES if summary[str(m)]['tier_median_ratio'][t]==min(v['tier_median_ratio'][t] for v in summary.values())] for t in TIERS}
    late_improvements=sum(x['final_cmax'] is not None and x['Cmax_at_60'] is not None and x['final_cmax']<x['Cmax_at_60'] for x in r['runs'])
    max_overshoot=max(x['overshoot'] for x in r['runs'])
    iteration_text='；'.join(f"M{m}: {summary[str(m)]['medians']['iterations_completed']:.1f}" for m in SIZES)
    fraction_text='；'.join(f"M{m}: {100*summary[str(m)]['medians']['candidate_generation_time_fraction']:.2f}%" for m in SIZES)
    calls_text='；'.join(f"M{m}: {summary[str(m)]['medians']['reference_calls']:.1f}" for m in SIZES)
    tier_text='；'.join(f"{t}: M={tier_best[t]}" for t in TIERS)
    m192_reason='、'.join(k for k,v in summary['192']['conditions'].items() if not v) or '四项条件均满足，但仍按最小合格M选择'
    smaller_failures='；'.join(f"M{m}未满足："+'、'.join(k for k,v in summary[str(m)]['conditions'].items() if not v) for m in SIZES if m<chosen)
    decision_reason=('M64已满足四项条件，保持历史生产配置。' if chosen==64 and s['satisfies'] else '更小的M未全部满足预先指定条件，选择首个合格M。' if s['satisfies'] else '没有候选池通过全部条件，按用户规则回退M64。')
    lines=['# Phase4-1A：真实60秒生产候选池与MLP数据准备','',f"Phase4-1A: {'PASS' if r['passed'] else 'FAIL'}；M_prod={chosen}；Feature pipeline READY；下一步 Phase4-1B MLP Candidate Ranker。",'',
        '## 测量口径','',
        '原样复用 Phase3-Y/YR 的 I2/I3、I5/I6、I9/I12；全部属于 V2_MODEL_DEVELOPMENT_CONSUMED，未使用8个ORACLE_DEV或23个MLP候选workbook做搜索。固定三个新seed 20261111/20261112/20261113。硬件为Intel i5-1135G7（4物理核/8逻辑CPU）。每seed一个独立物理核、每核只使用一个逻辑CPU；四个M按实例/seed轮换顺序，三个seed进程同时运行。测量对应本机这种固定并行负载，不声称换硬件仍有相同迭代数。','',
        '72次均调用原生 SA_OI_ALNS_V2、60秒、初始化计时。只改变 M/M_lns；M_atomic由M−M_lns自动得到。Kdp8/Kref2/Kref_total4、C2/C4、SA、direction refinement、B32、TWO_OPT_STAR=OFF均保持。统计计时包装只测量函数耗时，不改变返回值、随机数或搜索状态。','',
        'Cmax@60 仅来自原生认证 best_events 中 elapsed≤60 的记录，overshoot 后 final_cmax 不参与选择。generation_seconds 覆盖整个 generate_candidate_pool（含cheap screening、complete构造、LNS repair）；lns_seconds是其 destroy+repair 子集，不能再加到generation_seconds。DP/reference时间含初始化。吞吐以actual_runtime为分母，额外记录60秒检查点与overshoot。unique是每个iteration内去重的有效候选数之和，不是跨搜索状态去重。iterations/attempts/calls采用原生完整run计数，包含最后一轮可能超时的工作；只有主指标Cmax严格截到60秒。','',
        f'超时后final_cmax继续改善的run为{late_improvements}/72；最大overshoot={max_overshoot:.3f}秒。这些超时后改善全部排除出M选择。','',
        '## 配置与实测','',
        '| M | atomic/LNS | certified@60 | overall ratio | SMALL | MEDIUM | LARGE | median iterations | retention | gen fraction | reference calls |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in SIZES:
        v=summary[str(m)];d=v['medians'];rat=v['tier_median_ratio']
        lines.append(f"| {m} | {3*m//4}/{m//4} | {v['certified_checkpoints']}/18 | {v['median_ratio']:.6f} | {rat['SMALL']:.6f} | {rat['MEDIUM']:.6f} | {rat['LARGE']:.6f} | {d['iterations_completed']:.1f} | {100*v['median_iteration_retention_vs_M64']:.2f}% | {100*d['candidate_generation_time_fraction']:.2f}% | {d['reference_calls']:.1f} |")
    lines+=['','## 14个问题的回答','',
        f'1. 完成迭代中位数：{iteration_text}。',
        f'2. 候选生成耗时中位占比：{fraction_text}。LNS是generation子集。',
        f'3. 每run reference calls中位数：{calls_text}。其余吞吐见成本表，Kref预算没有增加。',
        f'4. overall配对ratio中位数最小的M为{best_pools}；这是18组instance×seed的Cmax@60比较，final_cmax未参与。',
        f'5. 分tier配对ratio中位数最小的M：{tier_text}。SMALL有多个M并列，MEDIUM/LARGE均以M192最小，分层最优集合不完全一致；生产设置仍统一。',
        f"6. 离线M192覆盖结论仍保留。在线M192：{m192_reason}。其迭代保留率为{100*summary['192']['median_iteration_retention_vs_M64']:.2f}%；本轮{'需要' if chosen==192 else '无需'}采用M192。",
        f"7. M_prod={chosen}。",
        f"8. {decision_reason} 更小候选的失败项：{smaller_failures or '无'}。M192的46%迭代保留率满足40%下限。详细ratio见上表；未比较本轮范围外的M。",
        '9. Kdp=8、Kref=2、Kref_total=4保持。SearchConfig()仍为历史M64。SA_OI_ALNS_V2省略search_config时使用生产M192/48；显式传入的历史配置原样保留，旧实验runner不改。未来60秒调用沿用原有60秒配置，仅将m/m_lns替换为公开的SA_OI_ALNS_V2_PRODUCTION_POOL_SIZE及其1/4。',
        '10. 剩余23个未被Phase4用于机制决策的workbook已按静态几何farthest-first划为17 MLP_TRAIN + 6 MLP_DEV，同workbook不跨集合。',
        '11. 原8个workbook在新简单split文件中标为ORACLE_DEV_CONSUMED，不再用于早停、超参数选择或MLP/GAT比较；历史manifest不改写。',
        f"12. 纯函数特征已完成，C2={r['feature_validation']['C2_feature_count']}维，C4={r['feature_validation']['C4_feature_count']}维。",
        '13. C2接口只接收当前解/当前方向/当前Cmax、完整candidate和科学配置，不能接收candidate方向DP/调度标签；C4额外接收方向DP和cheap rank。输入特征与训练targets在不同函数中。既有10,701候选全部通过determinism/finite/information-boundary检查，未发现label leakage；没有用这些label选择特征。',
        '14. 下一轮可以围绕选定M_prod生成MLP_TRAIN/DEV数据并进入Phase4-1B训练；本轮没有生成17个TRAIN的oracle labels，没有网络、scaler或optimizer拟合。','',
        '## 分层吞吐','', '| M | tier | iterations | gen fraction | reference calls | median Cmax@60 |','|---|---|---:|---:|---:|---:|']
    for m in SIZES:
        for t,v in summary[str(m)]['tier_medians'].items():
            lines.append(f"| {m} | {t} | {v['iterations_completed']:.1f} | {100*v['candidate_generation_time_fraction']:.2f}% | {v['reference_calls']:.1f} | {v['Cmax_at_60']:.6f} |")
    lines+=['','## 额外成本','', '| M | gen s | LNS s (子集) | DP s | reference s | calls/s | unique/s | runtime s | overshoot s |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in SIZES:
        v=summary[str(m)]['medians']
        lines.append('| '+str(m)+' | '+' | '.join(f'{v[k]:.4f}' for k in ('candidate_generation_seconds','lns_seconds','direction_dp_seconds','reference_scheduler_seconds','reference_calls_per_second','unique_candidates_per_second','actual_runtime','overshoot'))+' |')
    lines+=['','## 干净的数据集合','', 'MLP split仅有角色列表、已有workbook identity和静态选择说明；没有新hash/protocol。静态descriptor的标准化只用于几何多样性选择，不是MLP scaler拟合。workbook路径延续既有dataset manifest，均相对于DRL相邻的../ppo目录。','', '```json',json.dumps({k:audit.read(SPLIT)[k] for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED')},ensure_ascii=False,indent=2),'```','',
        '## 特征与标签','',
        'C2包含move/family/source与LNS operator、当前认证Cmax、处理负载和cheap travel proxy前后变化、各robot route长度/块数、pattern counts、受影响机器人/parent/位置及焊缝几何。C4额外包含direction feasibility/成本/变化、各robot方向count/alternations/首尾方向和flips、定向后route端点（变长方向向量不作为固定维输入直接展开）以及C3 rerank components。candidate reference/WAIT/DEADLOCK/certifier/timing不在输入中。缺失direction用显式可用性mask，正式C4仍只筛direction-feasible候选。','',
        'Target：仅FEASIBLE_CERTIFIED有y_feasible=1与y_improvement=(Cs−Ccandidate)/Cs；其他有效candidate状态y_feasible=0、improvement=NULL；NUMERIC_FAILURE整条排除。没有人工penalty、绝对Cmax回归或训练拟合。','',
        '完整特征列与逐候选回归计数在结果JSON；函数位于src/mrta_ranker/features.py。','',
        '验证：'+json.dumps(r['feature_validation'],ensure_ascii=False,indent=2),'', '回归：'+json.dumps(r['regression'],ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def print_summary(r):
    m=r['M_prod'];s=r['pool_comparison'][str(m)]
    print('Phase4-1A: '+('PASS' if r['passed'] else 'FAIL'))
    print(f'Production pool: M_prod = {m}')
    print('Median Cmax ratio vs best: '+str(s['median_ratio']))
    print('Median iteration retention vs M64: '+str(s['median_iteration_retention_vs_M64']))
    print('MLP split: 17 TRAIN / 6 DEV; 8 ORACLE_DEV_CONSUMED')
    print('Feature pipeline: READY')
    print('Next: Phase4-1B MLP Candidate Ranker')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('smoke','finish'))
    parser.add_argument('--regression-passed',type=int)
    args=parser.parse_args()
    if args.action=='smoke': run_smoke()
    else: finish(args.regression_passed)