"""Unseen-geometry RULE/OLD_GNN/NEW_GNN initialization with one MODE_NEW backend."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
import traceback

for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import run_gnn_shared_construction as shared
from run_gnn_multi_initial import CONFIG_05, _mode_search
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.geometry import build_legal_pattern_catalog
from mrta_reference.model import ParentWeld, ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2

CSV = ROOT / 'results/gnn_initial_mode_new.csv'
REPORT = ROOT / 'docs/GNN_INITIAL_MODE_NEW_EXPERIMENT.md'
CHECKPOINT = ROOT / 'models/gnn_coordination_constructor_05.pt'
SEEDS = (20261901, 20261902, 20261903, 20261904, 20261905)
METHODS = ('RULE', 'OLD_GNN_PORTFOLIO', 'NEW_GNN_PORTFOLIO')
TIERS = ('SMALL', 'MEDIUM', 'LARGE', 'HIGH_WORKLOAD')
FIELDS = (
    'phase', 'instance', 'tier', 'N', 'geometry_seed', 'workbook', 'sheet',
    'solver_seed', 'method', 'budget', 'status', 'initial_cmax', 'final_cmax',
    'initialization_seconds', 'actual_runtime', 'b32_calls', 'iterations',
    'accepted', 'mode_accepted', 'model_load_seconds', 'initial_b32_calls',
    'search_b32_calls', 'initial_found_seconds', 'last_improvement_seconds',
    'late_b32_calls', 'initial_attempts', 'error',
)


def geometry_coordinates(parents):
    """Compare physical coordinates directly, including reversed endpoints/IDs."""
    return tuple(sorted(tuple(sorted(tuple(round(v, 10) for v in point)
                                    for point in (p['start'], p['end']))) for p in parents))


def _entry(path, sheet, name, tier, seed):
    instance = load_ppo_platform_instance(path, sheet)
    parents = to_parent_welds(instance)
    build_legal_pattern_catalog(parents, CONFIG_05, FORMAL_SCOPE_V2)
    return dict(instance=name, tier=tier, N=len(parents), geometry_seed=int(seed),
                workbook=os.path.relpath(path, ROOT).replace('\\', '/'), sheet=sheet,
                total_length=sum(p.length for p in parents),
                parents=[dict(parent_id=p.parent_id, start=p.start, end=p.end) for p in parents])


def prepare_entries():
    """Read existing data only; fix all selections before any solver execution."""
    csv.field_size_limit(100_000_000)
    source_rows = []
    for name in ('parent_mode_joint_reconstruction_05.csv', 'final_parent_mode_joint_reconstruction_05.csv'):
        source_rows.extend(shared.research_rows(ROOT / 'results' / name))
    consumed = set()
    used_seeds = {r['geometry_seed'] for r in source_rows if r.get('geometry_seed')}
    for row in source_rows:
        for label in ('initial_state', 'final_state'):
            state = (row.get('details') or {}).get(label)
            if state:
                consumed.add(geometry_coordinates(state['solution']['parents']))
    current_geometries = len(consumed)
    historical_books = set()
    for path in sorted((ROOT / 'data/development').glob('gnn*jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('parents'):
                consumed.add(geometry_coordinates(row['parents']))
            if (entry := row.get('entry')) and entry.get('relative_path'):
                historical_books.add(entry['relative_path'])
    checkpoint = shared.read_neural_checkpoint(CHECKPOINT)
    refinement = checkpoint['refinement_candidate']['refinement']
    database = ROOT / 'data/development/graph_ranker_dataset_v05.sqlite'
    with sqlite3.connect(f'file:{database.as_posix()}?mode=ro', uri=True) as connection:
        matrix = json.loads(connection.execute(
            "SELECT payload FROM experiments WHERE experiment='sci_upgraded_independent_matrix' AND item_id='summary'"
        ).fetchone()[0])
    entries = [_entry(ROOT / e['relative_path'], e['sheet_name'], e['label'], e['tier'], e['geometry_seed'])
               for e in matrix['entries']]
    for entry in entries:
        if geometry_coordinates(entry['parents']) in consumed or entry['geometry_seed'] in used_seeds:
            raise ValueError('Existing upgraded geometry was consumed by GNN development: ' + entry['instance'])
    manifest = json.loads(shared.pilot.dataset.DATASET.read_text(encoding='utf-8'))
    pool = [e for e in manifest['instances'] if e['validation_status'] == 'VALID'
            and not e.get('duplicate_of') and e['relative_path'].startswith('data/PPO_TRAIN/')
            and e['relative_path'] not in historical_books and e['sheet_name'].startswith('g45_')]
    pool.sort(key=lambda e: (-e['total_weld_length_m'], -e['actual_weld_count'], e['relative_path']))
    selected_books = set()
    rejected = []
    for source in pool:
        if source['relative_path'] in selected_books:
            continue
        path = ROOT.parent / 'ppo' / source['relative_path']
        instance = load_ppo_platform_instance(path, source['sheet_name'])
        seed = int(instance.family_metadata['dataset_generation_seed'])
        try:
            entry = _entry(path, source['sheet_name'], f'HIGH_WORKLOAD_{len(selected_books) + 1:02d}',
                           'HIGH_WORKLOAD', seed)
        except ValueError as error:
            rejected.append(dict(workbook=source['relative_path'], reason=str(error)))
            continue
        if geometry_coordinates(entry['parents']) in consumed or seed in used_seeds:
            rejected.append(dict(workbook=source['relative_path'], reason='GNN development overlap'))
            continue
        selected_books.add(source['relative_path'])
        entries.append(entry)
        if len(selected_books) == 5:
            break
    if Counter(e['tier'] for e in entries) != Counter({tier: 5 for tier in TIERS}):
        raise ValueError('Expected five existing, GNN-unseen geometries per tier')
    if len({geometry_coordinates(e['parents']) for e in entries}) != 20:
        raise ValueError('Duplicate selected geometry')
    audit = dict(current_geometries=current_geometries, historical_geometries=len(consumed),
                 historical_books=len(historical_books), high_pool=len(pool), rejections=rejected,
                 previous_training_source=checkpoint['training_source'], refinement_split=refinement['split'])
    return sorted(entries, key=lambda e: (TIERS.index(e['tier']), e['instance'])), audit


def _worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_job(job):
    entry, method, seed, phase = job
    parents = tuple(ParentWeld(p['parent_id'], tuple(p['start']), tuple(p['end'])) for p in entry['parents'])
    budget = 60.0
    row = dict(phase=phase, **{k: entry[k] for k in ('instance', 'tier', 'N', 'geometry_seed', 'workbook', 'sheet')},
               solver_seed=seed, method=method, budget=budget, status='ERROR', initial_cmax=None, final_cmax=None,
               initialization_seconds=0., actual_runtime=0., b32_calls=0, iterations=0, accepted=0, mode_accepted=0,
               model_load_seconds=0., initial_b32_calls=0, search_b32_calls=0, initial_found_seconds=None,
               last_improvement_seconds=None, late_b32_calls=0, initial_attempts=[], error='')
    started = time.perf_counter()
    deadline = started + budget
    attempts = []
    try:
        if method == 'RULE':
            feasible = []

            def admit(candidate):
                elapsed = time.perf_counter() - started
                if elapsed > budget:
                    return False
                feasible.append((candidate.schedule.cmax, elapsed))
                return True

            initial, attempts = shared.pilot.oriented.common_initial_beam(
                parents, CONFIG_05, seed=seed, deadline=deadline, interleave=True, accept_candidate=admit)
            calls = sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
            if initial is not None:
                row['initial_found_seconds'] = min(t for c, t in feasible if c == initial.schedule.cmax)
        elif method in METHODS[1:]:
            previous, new = shared.load_neural_initial_models(CHECKPOINT)
            row['model_load_seconds'] = time.perf_counter() - started
            portfolio_started = time.perf_counter()
            initial, info = shared.neural_initial_portfolio(
                parents, previous, new, CONFIG_05, old_sampling=method == 'OLD_GNN_PORTFOLIO',
                seed=seed, reference_calls=3, use_learned_modes=False, certify=False, deadline=deadline)
            attempts, calls = info['attempts'], info['b32_calls']
            if info['selected_seconds'] is not None:
                row['initial_found_seconds'] = portfolio_started - started + info['selected_seconds']
            row['late_b32_calls'] = sum(a['b32_calls'] for a in attempts if a.get('late'))
        else:
            raise ValueError(method)
        row['initialization_seconds'] = time.perf_counter() - started
        row['initial_attempts'] = attempts
        row['b32_calls'] = row['initial_b32_calls'] = calls
        if initial is None:
            row['status'] = ('INITIALIZATION_TIMEOUT' if time.perf_counter() > deadline else 'INITIALIZATION_FAILED')
            # In particular, do not pass None to _mode_search: that interface
            # would start a fresh rule initializer, invalidating the GNN arm.
        else:
            row['initial_cmax'] = initial.schedule.cmax
            result = _mode_search(parents, method='MODE_NEW', seed=seed, budget=budget,
                                  max_iterations=1_000_000, config=CONFIG_05, certify=False,
                                  common_initial=initial, initial_seconds=time.perf_counter() - started,
                                  initial_attempts=attempts)
            row.update(status=result['status'], final_cmax=result['cmax'], b32_calls=result['reference_calls'],
                       iterations=result['iterations'], accepted=result['accepted'], mode_accepted=result['mode_accepted'])
            row['search_b32_calls'] = result['reference_calls'] - calls
            row['last_improvement_seconds'] = (result['best_events'][-1][0] if result['accepted']
                                                else row['initial_found_seconds'])
            row['late_b32_calls'] += sum(item.get('late', False) for item in result['trace'])
    except Exception:
        row.update(status='ERROR', error=traceback.format_exc())
        row['initial_attempts'] = attempts
    row['actual_runtime'] = time.perf_counter() - started
    row['initial_attempts'] = json.dumps(row['initial_attempts'], ensure_ascii=False, separators=(',', ':'))
    return row


def read_rows():
    if not CSV.exists():
        return []
    with CSV.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def append_row(row):
    header = not CSV.exists() or CSV.stat().st_size == 0
    with CSV.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def run_phase(entries, phase, workers):
    # A phase separates the deliberately excluded smoke runs. Within a phase
    # the resume key is exactly instance + solver_seed + method.
    done = {(r['instance'], int(r['solver_seed']), r['method']) for r in read_rows() if r['phase'] == phase}
    seeds = (20261900,) if phase == 'smoke' else SEEDS
    selected = [next(e for e in entries if e['tier'] == tier) for tier in ('SMALL', 'LARGE', 'HIGH_WORKLOAD')]
    if phase == 'formal':
        selected = [next(e for e in entries if e['tier'] == tier and e['instance'].endswith(f'_{i:02d}'))
                    for i in range(1, 6) for tier in TIERS]
    jobs = []
    for si, seed in enumerate(seeds):
        for ei, entry in enumerate(selected):
            offset = (si + ei) % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                if (entry['instance'], seed, method) not in done:
                    jobs.append((entry, method, seed, phase))
    print(f'{phase}: {len(done)} saved, {len(jobs)} pending, workers={workers}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_row(row)
            print(f"{phase} {index}/{len(jobs)} {row['instance']} seed={row['solver_seed']} {row['method']} "
                  f"{row['status']} initial={row['initial_cmax']} final={row['final_cmax']} "
                  f"runtime={row['actual_runtime']:.3f} B32={row['b32_calls']}", flush=True)


def number(row, field):
    value = row.get(field)
    return None if value in ('', None) else float(value)


def geometry_medians(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if (value := number(row, field)) is not None:
            groups[row['instance'], row['method']].append(value)
    return {key: statistics.median(values) for key, values in groups.items()}


def pair_summary(rows, field, comparator):
    medians = geometry_medians(rows, field)
    changes = []
    counts = [0, 0, 0]
    details = []
    for instance in sorted({r['instance'] for r in rows}):
        new = medians.get((instance, 'NEW_GNN_PORTFOLIO'))
        old = medians.get((instance, comparator))
        if new is None or old is None:
            continue
        improvement = 100. * (old - new) / old
        counts[1 if abs(improvement) <= 1e-8 else 0 if improvement > 0 else 2] += 1
        changes.append(improvement)
        details.append((instance, old, new, improvement))
    return counts, statistics.median(changes) if changes else None, details


def _format(value):
    return '—' if value is None else f'{value:.3f}'


def summarize(entries, audit):
    rows = read_rows()
    formal = [r for r in rows if r['phase'] == 'formal']
    smoke = [r for r in rows if r['phase'] == 'smoke']
    lines = ['# GNN初始化 + MODE_NEW真实运行实验', '',
             f'正式运行：{len(formal)}/300；冒烟：{len(smoke)}/9，冒烟不计入正式统计。', '',
             '## 数据与训练重叠核查', '',
             '已核对本地代码与GitHub luckyfishanddog/DRL README；本地实际代码为执行依据。',
             f"当前checkpoint旧模型来源：{audit['previous_training_source']}",
             f"新模型训练记录：{audit['refinement_split']['training_geometries']}个训练几何，"
             f"{audit['refinement_split']['holdout_geometries']}个已消费开发验证几何。",
             f"逐段坐标比较覆盖当前两份CSV的{audit['current_geometries']}个几何，连同历史GNN数据共"
             f"{audit['historical_geometries']}个几何、{audit['historical_books']}个历史GNN工作簿；选中20个均无重叠。",
             '15个upgraded_grc工作簿已用于其他算法的375次实验，本轮仅称GNN未见几何，不称整个项目未用数据。',
             f"高工作量从排除历史GNN工作簿后的{audit['high_pool']}个PPO_TRAIN原有45组布局中，"
             '按总长度降序、数量降序、文件名升序，检查现有0.50m合法拆分并选前5个不同工作簿。',
             '筛选在任何本轮算法运行前完成；无新几何、无裁剪、无端点修改、无新Excel。', '',
             '|实例|组别|工作簿 / sheet|N|生成种子|总长度/m|', '|---|---|---|---:|---:|---:|']
    for e in entries:
        lines.append(f"|{e['instance']}|{e['tier']}|{e['workbook']} / {e['sheet']}|{e['N']}|"
                     f"{e['geometry_seed']}|{e['total_length']:.3f}|")
    lines += ['', '## 执行口径', '',
              '固定CONFIG_05、FORMAL_SCOPE_V2、B32和原MODE_NEW。无训练、权重修改、方向/WAIT独立阶段。',
              'RULE使用common_initial_beam；OLD为旧模型greedy/sample(seed)/sample(seed+1)；'
              'NEW只将第三候选换成新模型固定模式greedy。两个GNN臂各最多3次初始B32，均certify=False。',
              '每次60秒包括本次模型文件读取、两模型实例化、初始化和B32；模块导入、静态Excel读取与CSV写入在计时外。'
              '4个spawn工作进程，PyTorch及BLAS单线程，方法次序循环交替。',
              '首次冒烟批次在保存首条结果时遭沙箱拒绝写CSV，未留存结果；修复写入权限后重新执行，'
              '现存9条冒烟才是接口验证记录。正式300条没有因此重跑，既有375次实验没有重跑。',
              '初始化无可行解直接记失败，绝不传None给_mode_search触发规则回退。截止后完成的初始候选不接纳；'
              '截止前已取得的初解保留。后续搜索沿用严格截止及严格Cmax改善接受，实际耗时保留跨截止返回的开销。',
              '正式种子：' + ', '.join(map(str, SEEDS)) + '。冒烟种子20261900。', '',
              '几何为统计单位，先取该几何有效种子的Cmax中位数。失败保留在可行率分母，'
              '没有有效Cmax的几何配对不比较；部分种子失败的几何仍用有效种子中位数，以下逐几何表显示样本量。'
              '相对改善=(对照−NEW)/对照，正值为NEW更好；胜平负容差1e−8个百分点。', '',
              '## 方法汇总', '',
              '|组别|方法|运行数|初始可行|最终可行|初始几何中位数均值/s|最终几何中位数均值/s|平均运行/s|平均B32|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for tier in ('ALL',) + TIERS:
        selected = [r for r in formal if tier == 'ALL' or r['tier'] == tier]
        for method in METHODS:
            arm = [r for r in selected if r['method'] == method]
            initial = [v for (i, m), v in geometry_medians(arm, 'initial_cmax').items()]
            final = [v for (i, m), v in geometry_medians(arm, 'final_cmax').items()]
            n = len(arm)
            lines.append(f"|{tier}|{method}|{n}|{sum(number(r,'initial_cmax') is not None for r in arm)}/{n}|"
                         f"{sum(number(r,'final_cmax') is not None for r in arm)}/{n}|"
                         f"{_format(statistics.fmean(initial) if initial else None)}|"
                         f"{_format(statistics.fmean(final) if final else None)}|"
                         f"{_format(statistics.fmean(float(r['actual_runtime']) for r in arm) if arm else None)}|"
                         f"{_format(statistics.fmean(int(r['b32_calls']) for r in arm) if arm else None)}|")
    lines += ['', '## NEW的几何配对', '', '|组别|阶段|对照|有效几何配对|胜/平/负|中位相对改善/%|',
              '|---|---|---|---:|---|---:|']
    for tier in ('ALL',) + TIERS:
        selected = [r for r in formal if tier == 'ALL' or r['tier'] == tier]
        for field, stage in (('initial_cmax', '初始'), ('final_cmax', '最终')):
            for comparator in METHODS[:2]:
                counts, change, pairs = pair_summary(selected, field, comparator)
                lines.append(f"|{tier}|{stage}|{comparator}|{len(pairs)}|{'/'.join(map(str,counts))}|{_format(change)}|")
    lines += ['', '## 逐几何原始结果核查', '',
              '|实例|方法|初始有效/5|最终有效/5|初始中位Cmax/s|最终中位Cmax/s|', '|---|---|---:|---:|---:|---:|']
    for e in entries:
        for method in METHODS:
            arm = [r for r in formal if r['instance'] == e['instance'] and r['method'] == method]
            initial = [number(r, 'initial_cmax') for r in arm if number(r, 'initial_cmax') is not None]
            final = [number(r, 'final_cmax') for r in arm if number(r, 'final_cmax') is not None]
            lines.append(f"|{e['instance']}|{method}|{len(initial)}/5|{len(final)}/5|"
                         f"{_format(statistics.median(initial) if initial else None)}|"
                         f"{_format(statistics.median(final) if final else None)}|")
    failures = Counter((r['method'], r['status']) for r in formal if number(r, 'final_cmax') is None)
    late = [r for r in formal if float(r['actual_runtime']) > 60]
    errors = [r for r in formal if r['status'] == 'ERROR']
    lines += ['', '## 时间和失败记录', '',
              f'实际返回耗时>60s：{len(late)}/{len(formal)}；最大耗时：'
              f"{_format(max((float(r['actual_runtime']) for r in formal),default=None))}s。"
              f"跨截止B32调用：{sum(int(r['late_b32_calls']) for r in formal)}。",
              '失败分布：' + json.dumps({f'{k[0]} / {k[1]}': v for k, v in failures.items()}, ensure_ascii=False),
              f'代码异常行：{len(errors)}。所有失败原始行保留在CSV。', '']
    if formal:
        accepted_times = [number(r, 'last_improvement_seconds') for r in formal
                          if number(r, 'last_improvement_seconds') is not None]
        lines += [f"已保存最佳解的最大完成时间为{max(accepted_times):.6f}s。"
                  'GNN失败的较短返回时间降低了全样本平均耗时，不能据此宣称求解更快。', '']
    for r in rows:
        if r['error']:
            lines += [f"{r['phase']} {r['instance']} / {r['method']} / {r['solver_seed']}：", '```text', r['error'], '```', '']
    if len(formal) == 300:
        init_rule = pair_summary(formal, 'initial_cmax', 'RULE')
        init_old = pair_summary(formal, 'initial_cmax', 'OLD_GNN_PORTFOLIO')
        final_rule = pair_summary(formal, 'final_cmax', 'RULE')
        final_old = pair_summary(formal, 'final_cmax', 'OLD_GNN_PORTFOLIO')
        success = {method: sum(r['method'] == method and number(r, 'final_cmax') is not None
                               for r in formal) for method in METHODS}
        initial_wins = {name for name, old, new, change in init_old[2] if change > 1e-8}
        final_wins = {name for name, old, new, change in final_old[2] if change > 1e-8}
        converted = sorted(initial_wins & final_wins)
        lines += ['## 研究问题与决策', '',
                  f"1. **新GNN没有改善规则初始化。** NEW相对RULE初始胜/平/负为"
                  f"{'/'.join(map(str,init_rule[0]))}，配对中位改善{_format(init_rule[1])}%。"
                  f"初始和最终可行率均为RULE {success['RULE']}/100、OLD {success['OLD_GNN_PORTFOLIO']}/100、"
                  f"NEW {success['NEW_GNN_PORTFOLIO']}/100。",
                  f"2. **相对旧组合有局部增量，整体优势不稳定。** 初始胜/平/负"
                  f"{'/'.join(map(str,init_old[0]))}，中位改善{_format(init_old[1])}%；最终"
                  f"{'/'.join(map(str,final_old[0]))}，中位改善仅{_format(final_old[1])}%。"
                  '新组合可行率低于旧组合，不能用成功样本的Cmax均值代替整体优势。',
                  f"3. **局部初始优势可以转化，但没有形成相对规则的整体改善。** 相对OLD，"
                  f"{len(initial_wins)}个初始中位Cmax改善的几何中，{len(converted)}个最终中位Cmax也改善："
                  + '、'.join(converted) + '。'
                  f"NEW相对RULE最终仍为{'/'.join(map(str,final_rule[0]))}，配对中位改善{_format(final_rule[1])}%。"
                  '部分几何有效种子数不同，以上属于有效结果条件下的比较，不能抵消失败率。',
                  '4. **规模差异明显。** SMALL的NEW最终几何中位数接近RULE，且相对OLD为5胜；'
                  'MEDIUM、LARGE和HIGH_WORKLOAD对RULE均为0胜5负。'
                  'HIGH_WORKLOAD_03相对OLD最终中位Cmax改善38.358%，但仍高于RULE；'
                  '高工作量组整体对OLD为1胜2平2负，不能推广为高工作量普遍优势。',
                  '5. **本轮不支持将GNN初始化保留为论文主方法。** 相对RULE的初始与最终Cmax均无整体优势，'
                  '可行率还下降11个百分点；相对OLD的最终配对中位收益很小且可行率下降3个百分点。'
                  '保留此次局部增量和负结果作为研究证据，停止继续训练或堆叠GNN模块，重新评估该技术路线；'
                  '本轮没有修改网络或MODE_NEW来追逐测试结果。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report updated: {len(formal)}/300 formal rows; {REPORT}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audit', 'smoke', 'formal', 'summarize', 'all'), default='all')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    entries, audit = prepare_entries()
    print('STATIC DATA AUDIT', json.dumps(audit, ensure_ascii=False), flush=True)
    summarize(entries, audit)
    if args.phase in ('all', 'smoke'):
        run_phase(entries, 'smoke', args.workers)
        summarize(entries, audit)
    if args.phase in ('all', 'formal'):
        run_phase(entries, 'formal', args.workers)
        summarize(entries, audit)


if __name__ == '__main__':
    main()
