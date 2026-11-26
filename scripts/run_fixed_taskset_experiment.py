"""Fixed Euclidean atom experiment; original budgets, data and B32 evaluator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS','1')

from mrta_reference.fixed_taskset import FixedTaskset, generate_taskset, expand_routes, job_span
from mrta_reference.model import ParentWeld, ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize, block_map
from mrta_reference.scheduler import reference_schedule_formal
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns, priority_schedule
from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga

CONFIG = ScientificConfig()
SEEDS = (20261110,20261111,20261112)
RESULTS = ROOT/'results/fixed_euclidean_split'
REPORT = ROOT/'docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md'
LIMIT = 100_000_000
FIELDS = ('split','layout_id','seed','method','status','cmax_s','native_cmax_s','runtime_s',
          'iterations','termination_reason','b32_calls','parameters','diagnostic','solution')
WAG_INPUT_ADAPTER = 'FIXED_ATOM_WAG_MERGED_ROUTE_NATIVE_FAST'
HGA_INPUT_ADAPTER = 'FIXED_HGA_ROUTE_CYCLE_VND_NATIVE_SELECT'


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',',':'))


def write_text(path, text):
    if len(text.encode('utf-8')) >= LIMIT:
        raise ValueError(f'file would exceed the user 100MB limit: {path}')
    path.write_text(text, encoding='utf-8')


def read_rows(method, *, include_earlier_adapters=False):
    path = RESULTS/f'{method}.csv'
    if not path.exists():
        return []
    csv.field_size_limit(LIMIT)
    import io
    text = path.read_text(encoding='utf-8')
    # WAG/HGA parents may append concurrently to different CSVs. Each JSON
    # field is escaped onto one physical line; ignore a still-writing tail.
    if text and not text.endswith('\n'):
        text = text.rsplit('\n',1)[0]+'\n'
    rows = list(csv.DictReader(io.StringIO(text)))
    if method in ('wag_vns','hga') and not include_earlier_adapters:
        adapter = WAG_INPUT_ADAPTER if method == 'wag_vns' else HGA_INPUT_ADAPTER
        rows = [r for r in rows if json.loads(r['diagnostic']).get('input_adapter') == adapter]
    return rows


def append_row(method, row):
    import io
    path = RESULTS/f'{method}.csv'
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    if not path.exists():
        writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if (path.stat().st_size if path.exists() else 0)+len(text.encode('utf-8')) >= LIMIT:
        raise ValueError(f'result file would exceed 100MB: {path}')
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text); stream.flush()


def prepare():
    path = RESULTS/'tasksets.json'
    if path.exists():
        print('FIXED_TASKSETS_EXIST', len(load_tasksets()), flush=True)
        return
    records, summaries = [], []
    # No alternative layout generation, no TRAIN-expanded or old task catalog.
    for split in ('train','validation','test'):
        data = ROOT/'data/weld_layout'/f'{split}.jsonl'
        for raw in map(json.loads, data.read_text(encoding='utf-8').splitlines()):
            parents = tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in raw['parents'])
            ts = generate_taskset(parents, CONFIG)  # A failure is fatal; no row is dropped.
            records.append(dict(split=split, layout_id=raw['layout_id'], **ts.record()))
            summaries.append(dict(split=split, layout_id=raw['layout_id'], **ts.statistics))
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_text(path, json_text(records))
    with (RESULTS/'taskset_summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=tuple(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    for split in ('train','validation','test'):
        rows = [s for s in summaries if s['split'] == split]
        totals = {k:sum(s[k] for s in rows) for k in rows[0] if k not in ('split','layout_id','shortest_task_m')}
        print('FIXED_TASKSET_SUMMARY',split,len(rows),json_text(totals),
              'shortest',min(s['shortest_task_m'] for s in rows),flush=True)
    update_report()


def load_tasksets():
    return [(row['split'],row['layout_id'],FixedTaskset.from_record(row))
            for row in json.loads((RESULTS/'tasksets.json').read_text(encoding='utf-8'))]


def solution_record(solution, directions, config=CONFIG):
    if solution is None or directions is None:
        return None
    blocks = block_map(solution,config)
    atoms = expand_routes(solution.routes)
    jobs = []
    for route in solution.routes:
        for identifier, bit in zip(route.block_ids,directions[route.robot_id]):
            b = blocks[identifier]
            jobs.append(dict(job_id=identifier,robot=route.robot_id,source_parent_id=b.parent_id,
                             u_start=b.u_start,u_end=b.u_end,start=b.start,end=b.end,length=b.length,direction=int(bit)))
    return dict(assignments={atom:r for r in range(4) for atom in atoms[r]},
                merged_jobs=jobs,routes=[list(route.block_ids) for route in solution.routes],
                directions=[list(map(int,bits)) for bits in directions])


def restore_solution(taskset, row):
    payload = json.loads(row['solution']) if isinstance(row['solution'],str) else row['solution']
    solution = canonicalize(taskset.parents,taskset.patterns,dict(enumerate(payload['routes'])),CONFIG,scope=FORMAL_SCOPE_V2)
    if {a:r for r,route in enumerate(expand_routes(solution.routes)) for a in route} != payload['assignments']:
        raise ValueError('saved assignment differs from merged route coverage')
    return solution,tuple(tuple(bits) for bits in payload['directions'])


def baseline_job(item, *, purpose='fixed_taskset_experiment'):
    from numba import set_num_threads
    set_num_threads(1)
    method,split,name,ts,seed = item
    print('FIXED_BASELINE_START',method,split,name,len(ts.tasks),seed,flush=True)
    started = time.perf_counter()
    settings = AdaptedWAGConfig() if method == 'wag_vns' else AdaptedHGAConfig(vnd_objective='ROUTE_CYCLE')
    run = run_adapted_wag_vns if method == 'wag_vns' else run_adapted_hga
    kwargs = dict(seed=seed,time_limit=math.inf,scope=FORMAL_SCOPE_V2,checkpoints=(),
                  certify=False,candidate_evaluator=priority_schedule,fixed_taskset=ts)
    row = dict(split=split,layout_id=name,seed=seed,method=method,parameters=json_text(asdict(settings)))
    try:
        result = run(ts.parents,CONFIG,settings,**kwargs)
        row.update(native_cmax_s=result.schedule.cmax if result.schedule and result.schedule.feasible else None,
                   iterations=result.iterations,termination_reason=result.termination_reason,b32_calls=0)
        expected = settings.imax if method == 'wag_vns' else settings.max_iterations
        if result.iterations != expected:
            row['status'] = 'INCOMPLETE'
        elif result.solution is None:
            row['status'] = result.status.value
        else:
            formal_started = time.perf_counter()
            schedule = reference_schedule_formal(result.solution,CONFIG,scope=FORMAL_SCOPE_V2,
                                                  orientations=dict(enumerate(result.directions)))
            formal_seconds = time.perf_counter()-formal_started
            row.update(status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,b32_calls=1,
                       solution=json_text(solution_record(result.solution,result.directions)))
        row['diagnostic'] = json_text(dict(search_accounting=result.accounting,
                search_diagnostics=result.diagnostics,search_runtime_s=result.actual_runtime,
                initialization_s=result.initialization_time,search_and_precompute_s=result.search_time,
                final_b32_s=0. if row['b32_calls']==0 else formal_seconds,
                formal_diagnostics=() if row['b32_calls'] == 0 else schedule.diagnostics))
    except Exception as error:
        import traceback
        row.update(status='ERROR',termination_reason='ERROR',diagnostic=json_text(dict(error=repr(error),traceback=traceback.format_exc())))
    if method in ('wag_vns','hga'):
        detail = json.loads(row['diagnostic'])
        detail['input_adapter'] = WAG_INPUT_ADAPTER if method == 'wag_vns' else HGA_INPUT_ADAPTER
        detail['purpose'] = purpose
        detail['single_thread'] = True
        detail['cold_jit_included'] = True
        if purpose == 'single_full_budget_timing':
            detail['runtime_context'] = os.environ.get('FIXED_EXPERIMENT_RUNTIME_CONTEXT','')
        row['diagnostic'] = json_text(detail)
    row['runtime_s'] = time.perf_counter()-started
    print('FIXED_BASELINE_RESULT',json_text({k:v for k,v in row.items() if k not in ('solution','diagnostic')}),flush=True)
    return row


def benchmark(methods, layout_id, seed):
    """Exactly one full run per method, serial, each in a fresh process."""
    if len(methods)>1:
        import subprocess
        for method in methods:
            subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),
                'benchmark','--method',method,'--layout',layout_id,'--seed',str(seed)],cwd=ROOT,check=True)
        return
    method = methods[0]
    tasksets = [(split,name,ts) for split,name,ts in load_tasksets() if name==layout_id]
    if len(tasksets)!=1 or tasksets[0][0]!='train':
        raise ValueError('the one-case timing run uses an existing TRAIN layout')
    previous = [r for r in read_rows(method) if r['layout_id']==layout_id and int(r['seed'])==seed
                and json.loads(r['diagnostic']).get('purpose')=='single_full_budget_timing'
                and r['termination_reason']=='ITERATION_LIMIT']
    if previous:
        print('FIXED_BENCHMARK_ALREADY_SAVED',method,json_text({k:previous[-1][k] for k in FIELDS if k not in ('solution','diagnostic')}),flush=True)
        return
    split,name,ts = tasksets[0]
    print('FIXED_BENCHMARK_INPUT',json_text(dict(layout_id=name,seed=seed,parents=len(ts.parents),
        atoms=len(ts.tasks),method=method,workers=1,threads=1,stop='500 VNS' if method=='wag_vns' else '200000 valid offspring')),
        flush=True)
    row = baseline_job((method,split,name,ts,seed),purpose='single_full_budget_timing')
    append_row(method,row); update_report()
    if row['termination_reason']!='ITERATION_LIMIT':
        raise RuntimeError(row['diagnostic'])


def baselines(methods=('wag_vns','hga'), splits=('train','validation','test'), workers=4):
    tasksets = load_tasksets()
    for method in methods:
        existing = read_rows(method)
        keys = [(r['split'],r['layout_id'],int(r['seed'])) for r in existing]
        if len(keys) != len(set(keys)):
            raise ValueError(f'duplicate result keys in {method}')
        required = 500 if method == 'wag_vns' else 200000
        done = {key for key,row in zip(keys,existing) if row['termination_reason'] == 'ITERATION_LIMIT'
                and int(row['iterations']) == required}
        failed = set(keys)-done
        if failed:
            raise RuntimeError(f'{method}: saved failed/incomplete runs require diagnosis before retry: {sorted(failed)}')
        pending = [(method,split,name,ts,seed) for split,name,ts in tasksets if split in splits
                   for seed in SEEDS if (split,name,seed) not in done]
        print('FIXED_BASELINE_PENDING',method,len(pending),'workers',workers,flush=True)
        if workers == 1:
            for item in pending:
                row = baseline_job(item); append_row(method,row); update_report()
                if row['termination_reason'] != 'ITERATION_LIMIT':
                    raise RuntimeError(f'baseline stopped before its required budget: {row["diagnostic"]}')
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(baseline_job,item):item for item in pending}
                for future in as_completed(futures):
                    row = future.result(); append_row(method,row); update_report()
                    if row['termination_reason'] != 'ITERATION_LIMIT':
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(f'baseline stopped before its required budget: {row["diagnostic"]}')


def update_report():
    tasksets = load_tasksets()
    earlier_wag = [r for r in read_rows('wag_vns',include_earlier_adapters=True)
                   if json.loads(r['diagnostic']).get('input_adapter') != WAG_INPUT_ADAPTER]
    lines = ['# 固定欧氏拆分任务集实验','',
      '当前请求：后台完成WAG/HGA各70排样×3种子的完整对照实验，再自动执行共享GNN的BC预训练8轮、PPO微调80次更新和最终三种子评价。WAG已按4个工作进程运行；HGA恢复旧正式入口后，先完成train_03/20261111的200000有效子代测速，无执行错误再加入4个工作进程批量。单例使用新Python进程、Numba/BLAS单线程，首次编译不扣除；HGA测速时WAG批量并行，运行上下文随测速记录保存。后续由本机后台进程顺序执行，不调用在线模型或使用模型轮询；实际完成情况以本报告表格、训练CSV及模型文件为准。',
      '本地工作目录为D:\\pybullet_test\\MRTA_GA\\DRL；指定解释器D:\\pybullet_test\\.venv\\Scripts\\python.exe。DRL没有独立.git，外围MRTA_GA实际HEAD为0935ad8a723057084be190e31c6be4733ce3cb62，外围remote是MRTA.git；未将其误记为独立DRL提交，也未覆盖或提交用户既有修改。',
      '',
      '数据及原始焊缝未修改。先判定整条焊缝的轨道资格；仅无法由任一轨道承担时，在共享区域内将参数0.5投影到合法区间，最大化较短段。优先两段严格大于1m，否则严格大于0.5m；不合法实例直接报告。区域段欧氏长度L>5m时按n=ceil(L)-1等分，每个新段严格大于1m。',
      '', '按原始参数相邻且同一机器人承担的任意多段，在排序及评价之前强制合并。方向作用于整个作业；加工时间20+L/0.0108+30秒，空走速度0.20m/s。最终使用原FORMAL_SCOPE_V2的B32，不新增调度器或认证器。',
      '', '|划分|排样|父焊缝|Y拆分|严格>1m|放宽>0.5m|长区域段|长拆分额外任务|原子任务|最短任务(m)|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('train','validation','test'):
        group = [ts.statistics for s,_,ts in tasksets if s == split]
        values = [sum(g[k] for g in group) for k in ('parent_count','y_split_count','y_strict_1m_count',
                  'y_relaxed_0_5m_count','long_region_count','long_split_extra_tasks','atomic_task_count')]
        lines.append('|'+ '|'.join(map(str,[split,len(group),*values,min(g['shortest_task_m'] for g in group)]))+'|')
    lines += ['', '原本未触发拆分的短焊缝原样保留；1m约束针对新生成的长焊缝子段。',
      '长焊缝拆分后子段总数（长区域段数＋额外任务数）：TRAIN 2888、VALIDATION 789、TEST 1375，共5052；表中的额外任务数明确不含被替换的区域段本身。',
      '长拆分继承Y阶段区域焊缝的轨道资格；不会因为某个短子段进入共享区而重新开放另一轨道。首次运行检查发现4个TRAIN子段资格需要这一修正，已停止未完成的旧资格计算并重启，未降低预算；此前已完成的train_01/20261111几何资格相同，其原记录保留，但算子适配版本按下文单独区分。',
      '', 'WAG参数：n=10、p=0.50、imax=500、factorial_window=5、PAPER_EXACT；不截断WAG变体或路线组合。HGA参数：mu=20、lambda=10、alpha=20、200000有效子代、完整六邻域、显式方向、ROUTE_CYCLE VND；固定任务域中禁用模式变异。HGA按旧正式入口在局部搜索后对每个完整个体调用一次原生协调评价，用协调Cmax进行选择及保留最优。最终一次B32。种子20261110/20261111/20261112；不使用墙钟终止。',
      '', 'HGA保留原交叉、alpha邻居和M1–M6。固定域数值表列举原子任务的可能连续并集；每次归属变化后重新合并。ROUTE_CYCLE局部搜索按合并后的路线加工及空走代价评价，不在邻域内部调用协调器。原双子段加工/路线剪枝对新并集不成立，固定域候选使用合并后的有效路线下界。',
      '慢速原因核查：旧scripts/run_gnn_weld_layout.py的formal HGA显式覆盖vnd_objective=ROUTE_CYCLE；新固定任务入口曾直接采用配置类默认COORDINATED，且错误地禁止固定域ROUTE_CYCLE，因而实际改变了搜索评价路径。原生内核虽保留，却被放进大量邻域候选内，不能据此称原快速路径未变。已恢复旧正式入口的ROUTE_CYCLE VND＋每完整个体一次原生协调选择＋最终B32；同时修正固定合并分支无条件协调的问题。旧正式CSV 90条完整200000运行中位1116.142秒（18.602分钟），新固定域耗时以本次完整实测为准。先前慢速尝试已停止，未以部分子代计为完整结果。',
      '固定域等价提速仍保留：仅可能产生新连续同机器人边界的动作重建并集，其他动作复用原区间评价；资格前缀提前排除不合法转移；去重表按当前有效作业计数分配，而非全部潜在并集。没有减少合法候选、改变接受顺序或增加VND截断。测试穷举合并后的六邻域，并检查ROUTE_CYCLE邻域内协调调用为零、每完整个体一次原生评价和最优时间线复用。',
      f'WAG历史诊断共{len(earlier_wag)}条均原样保留：初版原子算子曾绕开原生快速评价；随后按仓库整块算子测速得到87.861360秒，但不符合文献W_k/W\u2032_k的分层。两批都不与当前文献分层＋快速路径版本混用，CSV使用普通input_adapter来源字段区分，无新hash或manifest。',
      '随后按用户要求核对WAG原文，前一段“恢复仓库算子”的处理不能据此称为符合文献：Lee等论文第5849页算法1第5步在拆分集合S\u2032上生成未合并WAG W_k，第7–11步派生合并作业W\u2032_k，第12–19步在W\u2032_k上排序和调度，第21步保存best_WAG=W_k（已目视核对，无prime）。算法2第8/10/12步在best_WAG上生成邻域，第15步重新执行合并、排序和评价。按伪代码，分配/VNS层应保留拆分子边，排序/计时层使用合并作业；把合并作业永久当作不可拆的分配变量会缩小邻域。',
      '当前WAG：Move/Swap/LNS使用未合并固定子段归属W_k，每个候选先强制合并生成W\u2032_k，再调用原nearest/farthest、2-opt、factorial路线改善及原生协调评价。固定归属下改变路线顺序不会再次改变合并，故复用原数值快速路径及缓存，不反复展开/合并81个相同归属的路线组合。未删候选、未改变n/p/500轮、未新增方向模块。',
      '本地原文：references/Novel method for welding gantry robot scheduling at shipyards.pdf，第5847–5852页，第3.3、4.1、4.4节及算法1/2；DOI:10.1080/00207543.2022.2117869。原文为三机器人、水平投影长度拆分，LNS清空中间机器人并破坏两端p比例；本项目四机器人/欧氏拆分/B32属于指定模型适配，不能称全文逐项完全一致。',
      'HGA逐项核对Liu等论文第3.2.1–3.2.4及4.1节（本地references/1-s2.0-S0957417425009212-main.pdf，第6–7页）：固定域20个初始个体全部随机选任务，向最短合法路线作最小增量的联合位置/方向插入，随后完整VND；二元锦标赛、路线替换交叉、保护供体路线、删除重复、打乱并插回缺失任务、M1–M6、alpha=20及mu+lambda种群管理继续复用现有实现。固定任务域禁用模式变异。新增固定子段插入按合并后的真实作业计算增量：预计算连续并集、虚拟移除将被合并的旧作业，保留全部不同位置/方向结果，不反复物化全路线。穷举测试核对最短路线初始化和minmax修复两个准则。',
      'HGA原文没有本项目的固定拆分谱系和强制合并；其六邻域在当前实际连续作业上进行，交叉/缺失修复在原子覆盖层保持恰好一次。四机器人资格、同轨/跨轨干涉、20/30秒准备处理及原生协调选择是用户指定共同模型适配；不声称复现原文不同运动/甘特物理模型。现有Numba区间代价、精确下界、原生协调内核及VND缓存保留；COORDINATED可选路径及其前缀复用也保留，当前正式入口按旧实验使用ROUTE_CYCLE。',
      '', '|本轮单例测速|排样|种子|总耗时(s)|初始化(s)|搜索及预计算(s)|最终B32(s)|迭代数|B32状态|B32 Cmax(s)|',
      '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    timing_contexts = []
    for method in ('wag_vns','hga'):
        cases = [r for r in read_rows(method) if json.loads(r['diagnostic']).get('purpose')=='single_full_budget_timing']
        if not cases:
            lines.append(f'|{method}|train_03|20261111|||||尚未完成|||')
        for r in cases:
            d = json.loads(r['diagnostic'])
            values = [method,r['layout_id'],r['seed'],r['runtime_s'],d.get('initialization_s',''),
                      d.get('search_and_precompute_s',''),d.get('final_b32_s',''),r['iterations'],r['status'],r['cmax_s']]
            lines.append('|'+ '|'.join(map(str,values))+'|')
            if d.get('runtime_context'):
                timing_contexts.append(f'测速运行上下文（{method}）：{d["runtime_context"]}')
    lines += ['', *timing_contexts, '', '计时包含参数/表构造、冷JIT、初始化、全预算搜索及最终一次B32；读取既有任务集和新进程启动在单例计时之外。两方法分别启动新进程，不把前一算法的内存中JIT热身当成后一算法独立冷启动。尚未完成者不填估计耗时或Cmax。',
      '', '|算法|划分|已完成/计划|B32可行|可行率|平均Cmax(s)|中位Cmax(s)|平均耗时(s)|中位耗时(s)|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    all_rows = []
    for method in ('wag_vns','hga','gnn_ppo'):
        rows = read_rows(method); all_rows.extend(rows)
        for split,count in (('train',120),('validation',30),('test',60)):
            group = [r for r in rows if r['split'] == split]
            feasible = [float(r['cmax_s']) for r in group if r['status'] == 'FEASIBLE' and r['cmax_s']]
            durations = [float(r['runtime_s']) for r in group]
            values = [method,split,f'{len(group)}/{count}',len(feasible),
                      f'{len(feasible)/len(group):.2%}' if group else '尚未运行',
                      *[f'{f(feasible):.6f}' if feasible else '' for f in (mean,median)],
                      *[f'{f(durations):.3f}' if durations else '' for f in (mean,median)]]
            lines.append('|'+ '|'.join(map(str,values))+'|')
    lines += ['', '以上汇总仅包含本目录的新固定任务集实测。未落盘的运行不计入完成数；空值不代表零Cmax。每个算法结果CSV保存全部任务归属、合并作业、路线、方向、实际停止计数、参数及失败原因，直接作为TRAIN教师来源。',
              '', 'GNN设置：随机初始化共享两层64维/4头GAT Actor-Critic，完整TRAIN教师轨迹BC 8轮，随后PPO 80次更新、每次4条完整轨迹、4轮裁剪更新，clip=0.2、学习率0.0003、gamma=1、GAE lambda=0.95、熵系数0.01。唯一终止奖励为-B32 Cmax/固定TRAIN参考Cmax，不可行惩罚-10；无辅助奖励。每10次PPO更新评价全部VALIDATION及三个种子。',
              '有可行TRAIN教师时参考尺度使用最优教师B32 Cmax；没有可行BC标签的排样仍保留在PPO中，使用该实例固定加工Cmax下界作为尺度，来源随模型记录。TRAIN串行加工及最远空走时长上界/加工尺度最大为8.922938，因此-10对当前数据给予不可行候选明确劣势；此上界不用于搜索、剪枝或正式评价。',
              '', 'VALIDATION选择各阶段权重；PPO checkpoint明确保存selected_update（0表示VALIDATION仍选择BC初始策略）和实际最后训练参数，不把PPO未改善说成改善。TEST仅在最终模型训练结束后推理。训练配置与逐轮实际记录保存在gnn_ppo_training.csv；阶段权重分别为gnn_fixed_euclidean_bc.pt和gnn_fixed_euclidean_ppo.pt。',
              '', '当前复现命令（在DRL目录，WAG批量及HGA单例）：','', '```powershell',
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py baselines --method wag_vns --workers 4",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py benchmark --method hga --layout train_03 --seed 20261111",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py baselines --method hga --workers 4",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py train",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py compare",
              "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py summary",
              '```', '', '后台顺序：等待当前HGA单例进程正常结束，核对已保存测速，再运行HGA批量；等待当前WAG批量进程结束并续补未完成项；两算法各210条完整预算记录齐备后，才依次train（BC→PPO）、compare、summary。各阶段非零退出即停止后续训练并将异常写回本报告，不以部分预算充作完整结果。已有本版本同键完整结果只读取，不自动替换排样或降低预算。',
              '', '|算法|划分|排样|种子|状态|B32 Cmax(s)|耗时(s)|迭代数|停止原因|',
              '|---|---|---|---:|---|---:|---:|---:|---|']
    for r in sorted(all_rows,key=lambda r:(r['method'],r['split'],r['layout_id'],r['seed'])):
        lines.append('|'+ '|'.join(str(r.get(k,'')) for k in ('method','split','layout_id','seed','status',
                                                       'cmax_s','runtime_s','iterations','termination_reason'))+'|')
    for r in all_rows:
        if r['status'] != 'FEASIBLE':
            lines += ['',f"失败记录：{r['method']}/{r['layout_id']}/{r['seed']}：{r['diagnostic']}"]
    training = RESULTS/'gnn_ppo_training.csv'
    if training.exists():
        with training.open(encoding='utf-8',newline='') as stream: history = list(csv.DictReader(stream))
        lines += ['', '|GNN阶段|更新|BC损失|策略损失|Value损失|平均奖励|B32调用|耗时(s)|VALIDATION可行/总数|VALIDATION平均Cmax(s)|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in history:
            validation = json.loads(row['validation'])
            costs = [v['cmax_s'] for v in validation if v['status']=='FEASIBLE']
            line = [row.get(k,'') for k in ('stage','update','bc_loss','policy_loss','value_loss','mean_reward','b32_calls','runtime_s')]
            line += [f'{len(costs)}/{len(validation)}' if validation else '',mean(costs) if costs else '']
            lines.append('|'+ '|'.join(map(str,line))+'|')
    else:
        lines += ['', '**GNN实际训练尚未开始，模型权重尚未生成。** 需要先完成本次固定域WAG/HGA教师实验；单元测试中的微型BC/PPO执行不计为正式训练。']
    lines += ['', '2026-11-25本轮HGA完整回归155项通过，另12项针对性检查通过（有重叠）：包含固定合并后的六邻域穷举、完整重建与快速路径一致、ROUTE_CYCLE邻域内零协调调用、每完整个体一次原生协调评价及最优时间线复用。较早9项新增逻辑专项及核心回归313项亦已通过（有重叠），涵盖双阈值与欧氏拆分、资格继承、多段合并与一次准备处理、精确覆盖、GNN完整教师轨迹/真实B32、共享编码器BC和裁剪PPO实际更新与TRAIN/TEST隔离。真实TRAIN首个教师246个完整动作已只读重放通过，未因此提前优化GNN。',
              '旧GNN三模式特征及原数据类语义保留。宽范围兼容性检查另有3项旧断言未通过（CSV Mock接口、旧PPO checkpoint关系维度、历史Joint3精确数值）；没有将这些旧结果纳入本次固定任务域比较，也未据此调整本次GNN。']
    write_text(REPORT,'\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','baselines','benchmark','train','compare','summary','all'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--method',choices=('wag_vns','hga','both'),default='both')
    parser.add_argument('--split',choices=('train','validation','test','all'),default='all')
    parser.add_argument('--layout',default='train_03')
    parser.add_argument('--seed',type=int,default=20261111)
    args = parser.parse_args()
    if args.stage=='benchmark':
        benchmark(('wag_vns','hga') if args.method=='both' else (args.method,),args.layout,args.seed)
    if args.stage in ('prepare','all'): prepare()
    if args.stage in ('baselines','all'):
        baselines(('wag_vns','hga') if args.method == 'both' else (args.method,),
                  ('train','validation','test') if args.split == 'all' else (args.split,),args.workers)
    if args.stage in ('train','all'):
        from mrta_search.gnn_ppo import train_experiment
        train_experiment(sys.modules[__name__])
    if args.stage in ('compare','all'):
        from mrta_search.gnn_ppo import compare_experiment
        compare_experiment(sys.modules[__name__])
    if args.stage in ('summary','all'):
        update_report()
        print(REPORT.read_text(encoding='utf-8').split('复现命令')[0],flush=True)


if __name__ == '__main__':
    main()
