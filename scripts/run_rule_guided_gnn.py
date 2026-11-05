"""B32 exact acceleration and rule-guided GNN development experiments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import gzip
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import run_gnn_shared_construction as shared
import run_gnn_multi_initial as study
from mrta_reference.model import ParentWeld, SplitPattern, SplitKind, Rail, ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize, block_map
from mrta_reference.scheduler import SchedulerProfile, reference_schedule_formal
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds

CONFIG = study.CONFIG_05
CSV = ROOT / 'results/gnn_rule_guided.csv'
REPORT = ROOT / 'docs/GNN_RULE_GUIDED_EXPERIMENT.md'
INSTANCES = ROOT / 'data/gnn_rule_guided/instances.csv'
CHECKPOINT = ROOT / 'models/gnn_coordination_constructor_05.pt'
TRAINED_CHECKPOINT = ROOT / 'models/gnn_rule_guided_constructor_05.pt'
METHODS = ('RULE', 'CURRENT_GNN', 'MRV_GNN', 'RULE_GUIDED_GNN')
SEEDS = (20261901, 20261902, 20261903)
BINS = ((20, 30), (31, 50), (51, 70), (71, 90))
INSTANCE_FIELDS = ('instance', 'tier', 'purpose', 'workbook', 'sheet', 'N', 'geometry_seed', 'role')
FIELDS = ('phase', 'instance', 'tier', 'N', 'solver_seed', 'method', 'status', 'initial_cmax',
          'final_cmax', 'initialization_seconds', 'search_seconds', 'actual_runtime',
          'b32_calls', 'b32_seconds', 'iterations', 'pruned', 'accepted', 'mode_accepted', 'payload')


def read_results():
    csv.field_size_limit(100_000_000)
    if not CSV.exists():
        return []
    with CSV.open(encoding='utf-8-sig', newline='') as stream:
        return [dict(raw, payload=json.loads(raw['payload'])) for raw in csv.DictReader(stream)]


def save(row):
    header = not CSV.exists() or CSV.stat().st_size == 0
    raw = {k: row.get(k) for k in FIELDS}
    raw['payload'] = json.dumps(row.get('payload', {}), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    with CSV.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print('SAVED', row.get('phase'), row.get('instance'), row.get('method'), row.get('status'),
          row.get('final_cmax'), flush=True)


def restore(raw, directions):
    parents = tuple(ParentWeld(p['parent_id'], tuple(p['start']), tuple(p['end'])) for p in raw['parents'])
    patterns = tuple(SplitPattern(p['parent_id'], SplitKind(p['kind']), p['t'], p['point_id'],
                                 p['mandatory'], None if p['rail'] is None else Rail(p['rail']))
                     for p in raw['patterns'])
    routes = {int(r['robot_id']): tuple(r['block_ids']) for r in raw['routes']}
    solution = canonicalize(parents, patterns, routes, CONFIG, scope=FORMAL_SCOPE_V2,
                            revision=raw.get('revision', 0))
    if solution.patterns != patterns or any(solution.routes[r].block_ids != routes[r] for r in range(4)):
        raise ValueError('saved complete candidate changed under CONFIG_05')
    return solution, tuple(tuple(bits) for bits in directions)


def old_development_sources():
    rows = shared.research_rows(ROOT / 'results/parent_mode_joint_reconstruction_05.csv')
    sources = {}
    for row in rows:
        state = (row.get('details') or {}).get('initial_state')
        if state and row['instance'] not in sources:
            sources[row['instance']] = dict(instance=row['instance'], tier=row['tier'], N=row['N'], state=state)
    return sorted(sources.values(), key=lambda s: (s['N'], s['instance']))


def profile_cases():
    cases = []
    seen = set()
    for source in old_development_sources():
        state = source['state']
        solution, bits = restore(state['solution'], state['directions'])
        cases.append(dict(instance=source['instance'], expected='FEASIBLE', solution=asdict(solution), directions=bits))
        seen.add((solution.patterns, tuple(r.block_ids for r in solution.routes), bits))
    # Historical complete candidates are reused only after they satisfy the
    # current 0.50m canonical model. No Excel or sealed workbook is accessed.
    buckets = defaultdict(list)
    with gzip.open(study.STUDY_PATH, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record') != 'run' or row.get('version') != 'parent-mode-v4-outer' or not row.get('solution'):
                continue
            for trial in row.get('trace', []):
                if trial.get('status') != 'DEADLOCK' or not trial.get('routes'):
                    continue
                raw = dict(parents=row['solution']['parents'], patterns=trial['patterns'],
                           routes=[dict(robot_id=r, block_ids=ids) for r, ids in enumerate(trial['routes'])])
                try:
                    solution, bits = restore(raw, trial['directions'])
                except (ValueError, KeyError):
                    continue
                key = (solution.patterns, tuple(r.block_ids for r in solution.routes), bits)
                if key in seen:
                    continue
                seen.add(key)
                tier = 'SMALL' if len(solution.parents) <= 40 else 'MEDIUM' if len(solution.parents) <= 65 else 'LARGE'
                if len(buckets[tier]) < 4:
                    buckets[tier].append(dict(instance=row['instance'], expected='DEADLOCK',
                                              solution=asdict(solution), directions=bits))
            if all(len(buckets[t]) >= 4 for t in ('SMALL', 'MEDIUM', 'LARGE')):
                break
    for index in range(4):
        for tier in ('SMALL', 'MEDIUM', 'LARGE'):
            if index < len(buckets[tier]) and len(cases) < 20:
                cases.append(buckets[tier][index])
    if len(cases) != 20:
        raise ValueError(f'expected twenty current-valid saved candidates, found {len(cases)}')
    return cases


def run_profile():
    done = {(r['instance'], r['method']) for r in read_results() if r['phase'] == 'profile'}
    for index, case in enumerate(profile_cases(), 1):
        name = f"case_{index:02d}_{case['instance']}"
        solution, bits = restore(case['solution'], case['directions'])
        previous = None
        for repeat in range(3):
            method = f'B32_REPEAT_{repeat + 1}'
            if (name, method) in done:
                continue
            profile = SchedulerProfile()
            tick = time.perf_counter()
            result = reference_schedule_formal(solution, CONFIG, orientations=dict(enumerate(bits)),
                                               scope=FORMAL_SCOPE_V2, profile=profile)
            elapsed = time.perf_counter() - tick
            if previous is not None and asdict(result) != previous:
                raise ValueError('repeated B32 changed the complete schedule')
            previous = asdict(result)
            save(dict(phase='profile', instance=name, N=len(solution.parents), method=method,
                      status=result.status.value, final_cmax=result.cmax, actual_runtime=elapsed,
                      b32_calls=1, b32_seconds=elapsed,
                      payload=dict(profile=profile.as_dict(), baseline_deadlock=result.baseline_deadlock,
                                   recovery_rollouts=result.recovery_rollouts, source_expected=case['expected'],
                                   case=case if repeat == 0 else None)))


def run_profile_detail():
    from mrta_search.lns import oriented_route_state, oriented_route_lower_bound
    rows = read_results()
    done = {r['instance'] for r in rows if r['phase'] == 'profile_detail'}
    for source in rows:
        if source['phase'] != 'profile' or source['method'] != 'B32_REPEAT_1' or source['instance'] in done:
            continue
        case = source['payload']['case']
        solution, bits = restore(case['solution'], case['directions'])
        profile = SchedulerProfile()
        tick = time.perf_counter()
        result = reference_schedule_formal(solution, CONFIG, orientations=dict(enumerate(bits)),
                                           scope=FORMAL_SCOPE_V2, profile=profile)
        elapsed = time.perf_counter() - tick
        if result.status.value != source['status'] or (result.cmax is not None and result.cmax != float(source['final_cmax'])):
            raise ValueError('additional SchedulerProfile timers changed B32 output')
        state = oriented_route_state(tuple(r.block_ids for r in solution.routes), bits, block_map(solution, CONFIG), CONFIG)
        bound = oriented_route_lower_bound(state)
        if result.status is ScheduleStatus.FEASIBLE and bound > result.cmax + 1e-8:
            raise ValueError('P+MOVE bound exceeds actual B32 makespan')
        save(dict(phase='profile_detail', instance=source['instance'], N=len(solution.parents), method='B32_TIMERS',
                  status=result.status.value, final_cmax=result.cmax, actual_runtime=elapsed,
                  b32_calls=1, b32_seconds=elapsed,
                  payload=dict(profile=profile.as_dict(), baseline_deadlock=result.baseline_deadlock,
                               recovery_rollouts=result.recovery_rollouts, lower_bound=bound)))


def worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def measured_search(initial, seed, budget, iterations, prune, initial_seconds=0., attempts=()):
    original = shared.pilot.oriented.schedule_explicit
    stats = dict(calls=0, seconds=0.)
    def timed(*args, **kwargs):
        tick = time.perf_counter()
        stats['calls'] += 1
        try:
            return original(*args, **kwargs)
        finally:
            stats['seconds'] += time.perf_counter() - tick
    shared.pilot.oriented.schedule_explicit = timed
    try:
        out = study._mode_search(initial.solution.parents, 'MODE_NEW', seed, budget, iterations,
                                 config=CONFIG, certify=False, common_initial=initial,
                                 initial_seconds=initial_seconds, initial_attempts=attempts,
                                 prune_cmax_lb=prune)
    finally:
        shared.pilot.oriented.schedule_explicit = original
    return out, stats


def accepted_signature(out):
    return [dict(iteration=t['iteration'], patterns=t['patterns'], routes=t['routes'],
                 directions=t['directions'], cmax=t['cmax']) for t in out['trace'] if t['accepted']]


def benchmark_pair(job):
    source, seed, phase = job
    tick = time.perf_counter()
    solution, bits = restore(source['state']['solution'], source['state']['directions'])
    initial = shared.pilot.oriented.schedule_explicit(solution, bits, CONFIG)
    initial_seconds = time.perf_counter() - tick
    if initial.schedule.status is not ScheduleStatus.FEASIBLE:
        raise ValueError('saved initial is not B32 feasible')
    results = {}
    order = (False, True) if seed % 2 else (True, False)
    for prune in order:
        out, stats = measured_search(initial, seed, math.inf if phase == 'equivalence' else 60.,
                                     40 if phase == 'equivalence' else 1_000_000, prune,
                                     initial_seconds, [dict(status='FEASIBLE')])
        if stats['calls'] + 1 != out['reference_calls']:
            raise ValueError('B32 call accounting differs from actual calls')
        results[prune] = dict(out=out, stats=stats)
    old, new = results[False]['out'], results[True]['out']
    identical = (old['cmax'] == new['cmax'] and old['solution'] == new['solution']
                 and old['directions'] == new['directions'] and accepted_signature(old) == accepted_signature(new))
    if phase == 'equivalence':
        equivalent = identical and new['reference_calls'] <= old['reference_calls'] and old['iterations'] == new['iterations'] == 40
        status = 'EQUIVALENT' if equivalent else 'MISMATCH'
    else:
        status = 'FEASIBLE'
    arms = {}
    for prune, value in results.items():
        out, stats = value['out'], value['stats']
        arms['PRUNED' if prune else 'ORIGINAL'] = dict(
            cmax=out['cmax'], iterations=out['iterations'], calls=out['reference_calls'],
            b32_seconds=stats['seconds'], runtime=out['actual_runtime'], pruned=out['pruned_by_cmax_lb'],
            accepted=out['accepted'], mode_accepted=out['mode_accepted'],
            solution=out['solution'], directions=out['directions'], accepted_trace=accepted_signature(out),
            last_improvement_seconds=out['best_events'][-1][0])
    return [dict(phase=phase, instance=source['instance'], tier=source['tier'], N=source['N'],
                 solver_seed=seed, method='PRUNED_vs_ORIGINAL', status=status,
                 initial_cmax=initial.schedule.cmax, final_cmax=new['cmax'],
                 initialization_seconds=initial_seconds, actual_runtime=new['actual_runtime'],
                 b32_calls=new['reference_calls'], b32_seconds=results[True]['stats']['seconds'],
                 iterations=new['iterations'], pruned=new['pruned_by_cmax_lb'],
                 accepted=new['accepted'], mode_accepted=new['mode_accepted'],
                 payload=dict(arms=arms, identical=identical,
                              wall_clock_disabled=phase == 'equivalence', initial_replay_calls=1))]


def parallel_jobs(worker, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            for row in future.result():
                save(row)


def run_benchmark(phase, workers):
    sources = old_development_sources()
    if phase == 'time':
        sources = [s for tier in ('SMALL', 'MEDIUM', 'LARGE')
                   for s in [e for e in sources if e['tier'] == tier][:2]]
    done = {(r['instance'], int(r['solver_seed'])) for r in read_results() if r['phase'] == phase}
    jobs = [(source, seed, phase) for source in sources for seed in SEEDS[:2]
            if (source['instance'], seed) not in done]
    print(phase, len(jobs), 'pending paired jobs', flush=True)
    parallel_jobs(benchmark_pair, jobs, workers)


def data_catalog():
    manifest = json.loads((ROOT / 'data/manifests/PPO_DATASET_MANIFEST_V1.json').read_text(encoding='utf-8'))
    overlay = json.loads((ROOT / 'data/manifests/PPO_V2_DATA_ROLES_V1.json').read_text(encoding='utf-8'))
    roles = {e['relative_path']: e['new_v2_role'] for e in overlay['workbooks']}
    counts = defaultdict(Counter)
    usable = []
    for entry in manifest['instances']:
        role = roles[entry['relative_path']]
        counts[role]['sheets'] += 1
        if entry['validation_status'] == 'VALID' and not entry.get('duplicate_of'):
            counts[role]['valid_sheets'] += 1
            if role != 'ID_TEST_SEALED':
                usable.append(entry)
    return roles, usable, dict(counts)


def historical_gnn_books():
    books = set()
    for path in (ROOT / 'data/development').glob('gnn*jsonl'):
        for line in path.open(encoding='utf-8'):
            entry = json.loads(line).get('entry') or {}
            if entry.get('relative_path'):
                books.add(entry['relative_path'])
    # Only workbook identities, never algorithm scores, influence data selection.
    with (ROOT / 'results/gnn_initial_mode_new.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['workbook'].startswith('../ppo/'):
                books.add(row['workbook'][len('../ppo/'):])
    return books


def select_families(pool):
    """Two sheets per size band, in eight distinct workbook families."""
    options = {}
    for lo, hi in BINS:
        per_book = {}
        for entry in pool:
            if not lo <= entry['actual_weld_count'] <= hi:
                continue
            key = (abs(entry['actual_weld_count'] - (lo + hi) / 2), entry['sheet_name'])
            old = per_book.get(entry['relative_path'])
            if old is None or key < old[0]:
                per_book[entry['relative_path']] = (key, entry)
        options[lo, hi] = sorted((item[1] for item in per_book.values()),
                                 key=lambda e: (abs(e['actual_weld_count'] - (lo + hi) / 2), e['relative_path']))
    slots = [band for band in sorted(BINS, key=lambda band: len(options[band])) for _ in range(2)]
    def assign(index, used, selected):
        if index == len(slots):
            return selected
        band = slots[index]
        for entry in options[band]:
            if entry['relative_path'] not in used:
                found = assign(index + 1, used | {entry['relative_path']}, selected + [(band, entry)])
                if found is not None:
                    return found
        return None
    result = assign(0, set(), [])
    if result is None:
        raise ValueError('existing nonsealed data cannot supply eight distinct families across four bands')
    return result


def build_dataset():
    roles, usable, counts = data_catalog()
    consumed = historical_gnn_books()
    validation = select_families([e for e in usable if roles[e['relative_path']] == 'V2_VALIDATION'])
    holdout = select_families([e for e in usable if roles[e['relative_path']] == 'V2_TRAIN_POOL'
                              and e['relative_path'] not in consumed])
    held_books = {e['relative_path'] for _, e in holdout}
    train_pool = [e for e in usable if roles[e['relative_path']] == 'V2_TRAIN_POOL'
                  and e['relative_path'] not in held_books]
    train = []
    for lo, hi in BINS:
        candidates = [e for e in train_pool if lo <= e['actual_weld_count'] <= hi]
        candidates.sort(key=lambda e: (abs(e['actual_weld_count'] - (lo + hi) / 2), e['relative_path'], e['sheet_name']))
        # Prefer distinct workbooks within a band, then use distinct sheets if
        # necessary. Workbook families remain isolated across data purposes.
        chosen = []
        seen = set()
        for entry in candidates:
            if entry['relative_path'] not in seen:
                chosen.append(entry)
                seen.add(entry['relative_path'])
            if len(chosen) == 10:
                break
        for entry in candidates:
            if len(chosen) == 10:
                break
            if entry not in chosen:
                chosen.append(entry)
        if len(chosen) != 10:
            raise ValueError(f'insufficient existing train sheets N{lo}_{hi}')
        train.extend([((lo, hi), e) for e in chosen])
    rows = []
    for purpose, selected in (('train', train), ('validation', validation), ('holdout', holdout)):
        ordinals = Counter()
        for (lo, hi), entry in sorted(selected, key=lambda item: (item[0], item[1]['relative_path'], item[1]['sheet_name'])):
            tier = f'N{lo}_{hi}'
            ordinals[tier] += 1
            rows.append(dict(instance=f'{purpose.upper()}_{tier}_{ordinals[tier]:02d}', tier=tier, purpose=purpose,
                             workbook=entry['relative_path'], sheet=entry['sheet_name'], N=entry['actual_weld_count'],
                             geometry_seed=int(Path(entry['relative_path']).stem.split('_')[-1]),
                             role=roles[entry['relative_path']]))
    if INSTANCES.exists():
        existing = list(csv.DictReader(INSTANCES.open(encoding='utf-8')))
        if [{k: str(v) for k, v in r.items()} for r in rows] != existing:
            raise ValueError('existing selection differs; do not replace a consumed validation/holdout assignment')
    else:
        INSTANCES.parent.mkdir(parents=True, exist_ok=True)
        with INSTANCES.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=INSTANCE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if not any(r['phase'] == 'data_audit' for r in read_results()):
        save(dict(phase='data_audit', instance='PPO_DATA', method='STATIC_SELECTION', status='READY',
                  payload=dict(workbook_roles=dict(Counter(roles.values())), sheet_counts=counts,
                               selected_counts=dict(Counter(r['purpose'] for r in rows)),
                               historical_gnn_books=len(consumed), ppo_root=str(ppo_root()),
                               source='existing PPO only; metadata-selected holdout never opened; no new Excel')))
    return rows


def ppo_root():
    return Path(os.environ.get('MRTA_PPO_ROOT') or ROOT.parent / 'ppo').resolve()


def load_entry(entry):
    if entry['purpose'] == 'holdout' or entry['role'] == 'ID_TEST_SEALED':
        raise ValueError('holdout/sealed geometries are not development inputs')
    instance = load_ppo_platform_instance(ppo_root() / entry['workbook'], entry['sheet'], ppo_root=ppo_root())
    parents = to_parent_welds(instance)
    if len(parents) != int(entry['N']):
        raise ValueError('PPO workbook count differs from the selected sheet')
    return parents


def validation_worker(job):
    entry, method, seed, phase = job
    parents = load_entry(entry)
    row = dict(phase=phase, instance=entry['instance'], tier=entry['tier'], N=len(parents), solver_seed=seed,
               method=method, status='ERROR', initial_cmax=None, final_cmax=None, initialization_seconds=0.,
               search_seconds=0., actual_runtime=0., b32_calls=0, b32_seconds=0., iterations=0, pruned=0,
               accepted=0, mode_accepted=0)
    original = shared.pilot.oriented.schedule_explicit
    stats = dict(calls=0, seconds=0.)
    def timed(*args, **kwargs):
        tick = time.perf_counter()
        stats['calls'] += 1
        try:
            return original(*args, **kwargs)
        finally:
            stats['seconds'] += time.perf_counter() - tick
    shared.pilot.oriented.schedule_explicit = timed
    started = time.perf_counter()
    deadline = started + 60.
    attempts = []
    detail = dict(workbook=entry['workbook'], sheet=entry['sheet'], geometry_seed=entry['geometry_seed'],
                  model_load_seconds=0., error='', initial_found_seconds=None, last_improvement_seconds=None)
    try:
        if method == 'RULE':
            admitted = []
            def accept(candidate):
                elapsed = time.perf_counter() - started
                if elapsed > 60.:
                    return False
                admitted.append((candidate.schedule.cmax, elapsed))
                return True
            initial, attempts = shared.pilot.oriented.common_initial_beam(
                parents, CONFIG, seed=seed, deadline=deadline, accept_candidate=accept,
                interleave=True, prune_cmax_lb=True)
            if initial is not None:
                detail['initial_found_seconds'] = min(t for c, t in admitted if c == initial.schedule.cmax)
        else:
            checkpoint = TRAINED_CHECKPOINT if method == 'RULE_GUIDED_GNN_RETRAINED' else CHECKPOINT
            previous, new = shared.load_neural_initial_models(checkpoint)
            detail['model_load_seconds'] = time.perf_counter() - started
            portfolio_started = time.perf_counter()
            initial, info = shared.neural_initial_portfolio(
                parents, previous, new, CONFIG, seed=seed, reference_calls=3, use_learned_modes=False,
                certify=False, deadline=deadline, prune_cmax_lb=True,
                parent_selection_mode='learned' if method == 'CURRENT_GNN' else 'mrv',
                action_selection_mode='rule_guided' if method.startswith('RULE_GUIDED_GNN') else 'gnn')
            attempts = info['attempts']
            if info['selected_seconds'] is not None:
                detail['initial_found_seconds'] = portfolio_started - started + info['selected_seconds']
        row['initialization_seconds'] = time.perf_counter() - started
        detail.update(initial_b32_calls=stats['calls'], initial_b32_seconds=stats['seconds'], initial_attempts=attempts)
        if initial is None:
            row['status'] = 'INITIALIZATION_TIMEOUT' if time.perf_counter() > deadline else 'INITIALIZATION_FAILED'
        else:
            row['initial_cmax'] = initial.schedule.cmax
            out = study._mode_search(parents, 'MODE_NEW', seed, 60., 2 if phase == 'smoke' else 1_000_000,
                                     config=CONFIG, certify=False, common_initial=initial,
                                     initial_seconds=time.perf_counter() - started, initial_attempts=attempts,
                                     prune_cmax_lb=True)
            row.update(status=out['status'], final_cmax=out['cmax'], iterations=out['iterations'],
                       pruned=out['pruned_by_cmax_lb'], accepted=out['accepted'], mode_accepted=out['mode_accepted'])
            if out['reference_calls'] != stats['calls']:
                raise ValueError('actual B32 calls differ from backend count')
            detail['last_improvement_seconds'] = (out['best_events'][-1][0] if out['accepted']
                                                    else detail['initial_found_seconds'])
        detail['initial_attempts'] = attempts
        detail['search_b32_seconds'] = stats['seconds'] - detail['initial_b32_seconds']
        detail['search_b32_calls'] = stats['calls'] - detail['initial_b32_calls']
    except Exception:
        row['status'] = 'ERROR'
        detail['error'] = traceback.format_exc()
        detail['initial_attempts'] = attempts
    finally:
        shared.pilot.oriented.schedule_explicit = original
    row.update(actual_runtime=time.perf_counter() - started, b32_calls=stats['calls'], b32_seconds=stats['seconds'], payload=detail)
    row['search_seconds'] = max(0., row['actual_runtime'] - row['initialization_seconds'])
    return [row]


def run_validation(phase, workers):
    entries = build_dataset()
    selected = [e for e in entries if e['purpose'] == 'validation']
    if phase == 'smoke':
        selected = [selected[0], selected[-1]]
    done = {(r['instance'], int(r['solver_seed']), r['method']) for r in read_results() if r['phase'] == phase}
    jobs = []
    methods = ('RULE_GUIDED_GNN_RETRAINED',) if phase == 'retrained_validation' else METHODS
    for si, seed in enumerate((20261900,) if phase == 'smoke' else SEEDS):
        for ei, entry in enumerate(selected):
            offset = (si + ei) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                if (entry['instance'], seed, method) not in done:
                    jobs.append((entry, method, seed, phase))
    print(phase, len(jobs), f'pending runs; {len(methods)} methods, max three neural proposals', flush=True)
    parallel_jobs(validation_worker, jobs, workers)


def geometric_pair(rows, field, comparator, method='RULE_GUIDED_GNN'):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) not in ('', None):
            groups[row['instance'], row['method']].append(float(row[field]))
    counts = [0, 0, 0]
    changes = []
    for instance in sorted({r['instance'] for r in rows}):
        new, old = groups.get((instance, method)), groups.get((instance, comparator))
        if not new or not old:
            continue
        change = 100. * (statistics.median(old) - statistics.median(new)) / statistics.median(old)
        counts[1 if abs(change) <= 1e-8 else 0 if change > 0 else 2] += 1
        changes.append(change)
    return counts, statistics.median(changes) if changes else None


def initial_equivalence_worker(job):
    entry, method = job
    parents = load_entry(entry)
    previous, new = shared.load_neural_initial_models(CHECKPOINT) if method != 'RULE' else (None, None)
    outcomes = []
    for prune in (False, True):
        if method == 'RULE':
            initial, attempts = shared.pilot.oriented.common_initial_beam(parents, CONFIG, seed=20261900,
                                                                         interleave=True, prune_cmax_lb=prune)
            calls = sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
        else:
            initial, info = shared.neural_initial_portfolio(parents, previous, new, CONFIG, seed=20261900,
                reference_calls=3, use_learned_modes=False, certify=False, prune_cmax_lb=prune,
                parent_selection_mode='learned' if method == 'CURRENT_GNN' else 'mrv',
                action_selection_mode='rule_guided' if method == 'RULE_GUIDED_GNN' else 'gnn')
            attempts, calls = info['attempts'], info['b32_calls']
        outcomes.append(dict(cmax=None if initial is None else initial.schedule.cmax, calls=calls,
                             solution=None if initial is None else asdict(initial.solution),
                             directions=None if initial is None else initial.directions,
                             pruned=sum(a['status'] == 'PRUNED_BY_CMAX_LB' for a in attempts)))
    a, b = outcomes
    same = all(a[k] == b[k] for k in ('cmax','solution','directions')) and b['calls'] <= a['calls']
    return [dict(phase='initial_equivalence', instance=entry['instance'], tier=entry['tier'], N=len(parents),
                 solver_seed=20261900, method=method, status='EQUIVALENT' if same else 'MISMATCH',
                 final_cmax=b['cmax'], b32_calls=b['calls'], pruned=b['pruned'],
                 payload=dict(original=a, pruned=b, wall_clock_disabled=True))]


def run_initial_equivalence(workers):
    selected = [e for e in build_dataset() if e['purpose'] == 'validation']
    entries = [next(e for e in selected if e['tier'] == f'N{lo}_{hi}') for lo, hi in BINS]
    done = {(r['instance'], r['method']) for r in read_results() if r['phase'] == 'initial_equivalence'}
    jobs = [(e, method) for e in entries for method in METHODS if (e['instance'], method) not in done]
    parallel_jobs(initial_equivalence_worker, jobs, workers)


def geometry_key(parents):
    return tuple(sorted(tuple(sorted((tuple(round(v, 10) for v in p.start),
                                     tuple(round(v, 10) for v in p.end)))) for p in parents))


def legacy_training_rows():
    checkpoint = shared.read_neural_checkpoint(CHECKPOINT)
    excluded = set(checkpoint['refinement_candidate']['refinement']['split']['holdout_names'])
    rows = shared.research_rows(ROOT / 'results/parent_mode_joint_reconstruction_05.csv')
    rows += shared.research_rows(ROOT / 'results/final_parent_mode_joint_reconstruction_05.csv')
    return [r for r in rows if r['instance'] not in excluded and r['status'] in ('FEASIBLE', 'CERTIFIED')]


def teacher_worker(entry):
    parents = load_entry(entry)
    started = time.perf_counter()
    deadline = started + 60.
    original = shared.pilot.oriented.schedule_explicit
    stats = dict(calls=0, seconds=0.)
    def timed(*args, **kwargs):
        tick = time.perf_counter(); stats['calls'] += 1
        try:
            return original(*args, **kwargs)
        finally:
            stats['seconds'] += time.perf_counter() - tick
    shared.pilot.oriented.schedule_explicit = timed
    row = dict(phase='teacher', instance=entry['instance'], tier=entry['tier'], N=len(parents),
               solver_seed=SEEDS[0], method='RULE_MODE_NEW', status='ERROR', initial_cmax=None, final_cmax=None,
               iterations=0, accepted=0, mode_accepted=0, pruned=0, initialization_seconds=0.)
    payload = dict(states=[], error='', workbook=entry['workbook'], sheet=entry['sheet'])
    try:
        initial, attempts = shared.pilot.oriented.common_initial_beam(parents, CONFIG, seed=SEEDS[0],
            deadline=deadline, interleave=True, prune_cmax_lb=True,
            accept_candidate=lambda candidate: time.perf_counter() <= deadline)
        row['initialization_seconds'] = time.perf_counter() - started
        if initial is None:
            row['status'] = 'INITIALIZATION_FAILED'
            payload['initial_attempts'] = attempts
        else:
            payload['states'].append(dict(method='RULE', state=study._final_state(
                initial.solution, initial.directions, initial.schedule, CONFIG)))
            row['initial_cmax'] = initial.schedule.cmax
            out = study._mode_search(parents, 'MODE_NEW', SEEDS[0], 60., 1_000_000,
                config=CONFIG, certify=False, common_initial=initial, initial_seconds=time.perf_counter()-started,
                initial_attempts=attempts, save_timeline=True, prune_cmax_lb=True)
            payload['states'].append(dict(method='MODE_NEW', state=out['final_state']))
            row.update(status=out['status'], final_cmax=out['cmax'], iterations=out['iterations'],
                       accepted=out['accepted'], mode_accepted=out['mode_accepted'], pruned=out['pruned_by_cmax_lb'])
    except Exception:
        payload['error'] = traceback.format_exc()
    finally:
        shared.pilot.oriented.schedule_explicit = original
    row.update(actual_runtime=time.perf_counter()-started, b32_calls=stats['calls'], b32_seconds=stats['seconds'], payload=payload)
    row['search_seconds'] = max(0., row['actual_runtime'] - row['initialization_seconds'])
    return [row]


def fallback_teacher_worker(entry):
    parents = load_entry(entry)
    rows = []
    for method in ('WAG', 'HGA'):
        out = study._final_baseline(parents, method, SEEDS[0], 60.)
        states = [dict(method=method, state=out['final_state'])] if out.get('final_state') else []
        rows.append(dict(phase='teacher_fallback', instance=entry['instance'], tier=entry['tier'], N=len(parents),
                         solver_seed=SEEDS[0], method=method, status=out['status'], final_cmax=out['cmax'],
                         actual_runtime=out['actual_runtime'], b32_calls=out['reference_calls'], iterations=out['iterations'],
                         payload=dict(states=states, error='')))
        if states:
            break
    return rows


def run_teachers(workers):
    entries = [e for e in build_dataset() if e['purpose'] == 'train']
    old_teachers, old_coverage = shared.research_teachers(legacy_training_rows(), CONFIG, deployment_only=False)
    old_geometries = {geometry_key(t['solution'].parents) for t in old_teachers}
    seen = set()
    for entry in entries:
        key = geometry_key(load_entry(entry))
        if key in seen or key in old_geometries:
            raise ValueError('selected PPO training sheet duplicates retained training geometry')
        seen.add(key)
    existing = read_results()
    if not any(r['phase'] == 'teacher_plan' for r in existing):
        save(dict(phase='teacher_plan', instance='PPO_TRAIN', method='RETAIN_AND_EXPAND', status='READY',
                  payload=dict(retained_geometries=len(old_geometries), retained_teachers=len(old_teachers),
                               additional_geometries=len(seen), duplicate_geometries=0, old_coverage=old_coverage,
                               policy='reuse prior 21-geometry teachers; generate RULE and MODE_NEW once per new sheet; '
                                      'WAG/HGA unchanged only if no executable teacher; no certifier or duplicate scheduling')))
    done = {r['instance'] for r in existing if r['phase'] == 'teacher'}
    parallel_jobs(teacher_worker, [e for e in entries if e['instance'] not in done], workers)
    rows = read_results()
    have = {r['instance'] for r in rows if r['phase'] in ('teacher','teacher_fallback') and r['payload'].get('states')}
    fallback_done = {r['instance'] for r in rows if r['phase'] == 'teacher_fallback'}
    missing = [e for e in entries if e['instance'] not in have and e['instance'] not in fallback_done]
    if missing:
        parallel_jobs(fallback_teacher_worker, missing, workers)


def run_training():
    from types import SimpleNamespace
    rows = legacy_training_rows()
    train_names = {e['instance'] for e in build_dataset() if e['purpose'] == 'train'}
    generated = read_results()
    new_names = set()
    for row in generated:
        if row['phase'] not in ('teacher','teacher_fallback'):
            continue
        if row['instance'] not in train_names:
            raise ValueError('nontraining geometry appeared in the teacher collection')
        for saved in row['payload'].get('states', []):
            new_names.add(row['instance'])
            rows.append(dict(instance=row['instance'], method=saved['method'], seed=int(row['solver_seed']),
                             status='FEASIBLE', details=dict(final_state=saved['state'])))
    teachers, coverage = shared.research_teachers(rows, CONFIG, deployment_only=False)
    for teacher in teachers:
        teacher['b32_status'] = 'FEASIBLE'
    if TRAINED_CHECKPOINT.exists():
        raise FileExistsError('new checkpoint already exists; do not overwrite or retrain it')
    _, model = shared.load_neural_initial_models(CHECKPOINT)
    graphs, examples, steps, replays = shared.research_prepare(teachers, CONFIG, certify=False)
    bounds = {}
    for teacher in teachers:
        key = teacher['geometry_key']
        bounds[key] = min(bounds.get(key, math.inf), teacher['cmax'])
    model, training = shared.research_train(graphs, examples, steps, SimpleNamespace(research_epochs=20),
                                           model=model, geometry_bounds=bounds)
    original = shared.read_neural_checkpoint(CHECKPOINT, include_research=False)
    keep = ('state_dict','model_type','hidden_dim','heads','normalize_inputs','use_dynamic_context','scientific_config')
    payload = {k: original[k] for k in keep if k in original}
    candidate = {k: original['refinement_candidate'][k] for k in keep if k in original['refinement_candidate']}
    candidate['state_dict'] = model.state_dict()
    candidate['use_dynamic_context'] = False
    candidate['refinement'] = dict(training=training, coverage=coverage, replays=replays,
        retained_geometry_count=21, additional_teacher_geometry_count=len(new_names),
        teacher_source='B32 FEASIBLE states retained in results/gnn_rule_guided.csv and original 21-geometry training rows',
        inference=dict(parent_selection_mode='mrv',action_selection_mode='rule_guided',use_learned_modes=False),
        holdout='8 reserved PPO sheets never opened; no validation teacher or normalization input')
    payload['refinement_candidate'] = candidate
    with TRAINED_CHECKPOINT.open('xb') as stream:
        shared.save_neural_checkpoint(payload, stream)
    save(dict(phase='training', instance='EXPANDED_TRAIN', method='SAME_STRUCTURE_20_EPOCHS', status='TRAINED',
              actual_runtime=training['runtime_s'], payload=dict(training=training, coverage=coverage,
                new_teacher_geometries=len(new_names), replayed_teachers=sum(r['replay_status']=='REPLAYED' for r in replays),
                failed_replays=sum(r['replay_status']!='REPLAYED' for r in replays), checkpoint=str(TRAINED_CHECKPOINT))))


def report():
    rows = read_results()
    profiles = [r for r in rows if r['phase'] == 'profile']
    lines = ['# B32安全剪枝与规则引导GNN实验', '', '## 实施边界与剪枝依据', '',
             '执行环境为D:\\pybullet_test\\.venv\\Scripts\\python.exe，工作目录为项目DRL。'
             '固定四机器人双轨、min_child_length=0.50m、delta_x=delta_y=0.50m、FORMAL_SCOPE_V2和B32最多32次恢复rollout。'
             'MODE_NEW的父选择、模式调整、方向联合插入和严格Cmax接受不变，WAG/HGA代码不修改。',
             '对当前完整路线及方向，LB=max_r(P_r+M_r)。P由config.process_time累计SETUP/WELD/POST，'
             'M为当前方向下的相邻焊缝空走；既有开放路线不计首任务进场或返家。B32只增加非负等待，故LB≤真实Cmax。',
             '搜索沿用epsilon=1e−9×max(1,Cmax_best)，LB≥Cmax_best−epsilon时跳过B32。'
             '初始化原来按精确最小Cmax选择，因此采用更保守的LB≥Cmax_best，保留小于搜索epsilon的初始化改善。'
             '没有可行初解时继续B32；跳过项标记PRUNED_BY_CMAX_LB，Cmax留空。',
             '实现复用OrientedRouteState.loads和arc_costs；未使用round后的oriented_time_score或exposure_score。'
             'RepairResult/JointCompletionResult保留已有路线状态以避免重复计算。报告数值为显示取整，CSV保留原始精度。', '',
             '主要修改位于src/mrta_search/lns.py、src/mrta_search/gnn_constructor.py、'
             'scripts/run_gnn_multi_initial.py、scripts/run_joint_oriented_repair_pilot.py和scripts/run_gnn_shared_construction.py。'
             'scheduler.py只为既有SchedulerProfile补充基准/完整恢复计时，PreparedDispatchProblem缓存与正式调度策略保留。', '',
             '## B32固定候选性能测量', '',
             '沿用SchedulerProfile和正式FORMAL_SCOPE_V2；相同候选重复3次，调度结果逐项比较。',
             '基准调度正常结束时不执行恢复；只有基准DEADLOCK才进入最多32个rollout的既有恢复。', '',
             '|状态|调度数|平均总秒|路线/模板秒|主循环秒|ESS秒|干涉区间秒|冲突验证秒|恢复rollout秒|平均rollout数|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for status in sorted({r['status'] for r in profiles}):
        arm = [r for r in profiles if r['status'] == status]
        def avg(key):
            return statistics.fmean(r['payload']['profile'][key] for r in arm)
        lines.append(f"|{status}|{len(arm)}|{statistics.fmean(float(r['actual_runtime']) for r in arm):.6f}|"
                     + '|'.join(f'{avg(k):.6f}' for k in ('route_template', 'scheduler_main_loop', 'ESS',
                         'forbidden_interval', 'conflict', 'recovery_rollout_time', 'recovery_rollout_count')) + '|')
    lines += ['', '各计时项有嵌套关系，不能相加；恢复rollout计时不包含整个恢复前沿管理。', '']
    detail = [r for r in rows if r['phase'] == 'profile_detail']
    if detail:
        p = [r['payload']['profile'] for r in detail]
        lines += ['在同一20个候选上给既有SchedulerProfile补充两个非决策计时字段，直接测量基准调度和整个恢复调用：', '',
                  f"平均基准调度{statistics.fmean(x['baseline_dispatch'] for x in p):.6f}s；"
                  f"整个恢复{statistics.fmean(x['recovery'] for x in p):.6f}s；"
                  f"总返回{statistics.fmean(float(r['actual_runtime']) for r in detail):.6f}s。"
                  '状态和Cmax与原测量一致。模板占比很小，故未新增模板缓存或削减恢复。', '']
    for phase, title in (('equivalence', '固定40迭代等价验证'), ('time', '相同60秒预算比较')):
        selected = [r for r in rows if r['phase'] == phase]
        lines += ['## ' + title, '', '|实例|种子|状态|原始/剪枝Cmax|原始/剪枝B32计次（含初解）|原始/剪枝迭代数|剪枝数|',
                  '|---|---:|---|---|---|---|---:|']
        for row in selected:
            a, b = row['payload']['arms']['ORIGINAL'], row['payload']['arms']['PRUNED']
            lines.append(f"|{row['instance']}|{row['solver_seed']}|{row['status']}|{a['cmax']:.6f}/{b['cmax']:.6f}|"
                         f"{a['calls']}/{b['calls']}|{a['iterations']}/{b['iterations']}|{b['pruned']}|")
        lines += ['', '两臂复用同一个保存的可行初解，重放及一次B32耗时等额计入各臂。'
                  '固定迭代使用无限墙钟预算；比较最终完整方案、方向和逐次接受方案/顺序。'
                  '60秒比较允许完成不同迭代数，但仍拒绝截止后改善。', '']
    equivalent = [r for r in rows if r['phase'] == 'equivalence']
    timed = [r['payload']['arms'] for r in rows if r['phase'] == 'time']
    if len(equivalent) == 18 and len(timed) == 12:
        old_calls = sum(r['payload']['arms']['ORIGINAL']['calls'] for r in equivalent)
        new_calls = sum(r['payload']['arms']['PRUNED']['calls'] for r in equivalent)
        replay_calls = len(equivalent)
        old_search_calls, new_search_calls = old_calls - replay_calls, new_calls - replay_calls
        wins = sum(p['PRUNED']['cmax'] < p['ORIGINAL']['cmax'] - 1e-7 for p in timed)
        ties = sum(abs(p['PRUNED']['cmax'] - p['ORIGINAL']['cmax']) <= 1e-7 for p in timed)
        ratio = statistics.median(p['PRUNED']['iterations'] / p['ORIGINAL']['iterations'] for p in timed)
        lines += [f'固定工作量18组全部等价，搜索阶段B32实际调用{old_search_calls}→{new_search_calls}，减少'
                  f'{100*(old_search_calls-new_search_calls)/old_search_calls:.3f}%。'
                  f'初解每对物理重放一次，共{replay_calls}次；为两臂等额计入后，表中计次为{old_calls}→{new_calls}。'
                  f'60秒12组为{wins}胜/{ties}平/{12-wins-ties}负，迭代数比值中位数{ratio:.3f}倍。'
                  '12组来自6个几何×2种子，是性能重复测量，不按12个独立几何作统计推断。', '']
    data = [r for r in rows if r['phase'] == 'data_audit']
    if data:
        info = data[-1]['payload']
        lines += ['## PPO数据审查和扩充', '',
                  'MRTA_PPO_ROOT未设置时使用项目ppo目录。数据用途由既有角色文件决定，不能按物理目录名判断。',
                  '未打开ID_TEST_SEALED工作簿；保留holdout只从未被历史GNN和上一轮正式比较消费的V2_TRAIN_POOL工作簿中静态选择，'
                  '其Excel不载入、不运行算法、不用于模型选择。训练、验证、holdout按工作簿隔离。', '',
                  '|既有角色|工作簿数|sheet数|有效唯一sheet数|', '|---|---:|---:|---:|']
        for role, count in info['workbook_roles'].items():
            detail = info['sheet_counts'][role]
            lines.append(f"|{role}|{count}|{detail['sheets']}|{detail['valid_sheets']}|")
        lines += ['', '已有数据已满足40训练、8验证、8保留测试的四数量段目标，无需新Excel。'
                  '实例清单为data/gnn_rule_guided/instances.csv；训练集不同sheet可能属于同一工作簿，'
                  '因此不能把共享工作簿前缀视为独立测试家族。', '']
    validation = [r for r in rows if r['phase'] == 'validation']
    retrained = [r for r in rows if r['phase'] == 'retrained_validation']
    all_validation = validation + retrained
    display_methods = METHODS + (('RULE_GUIDED_GNN_RETRAINED',) if retrained else ())
    smoke = [r for r in rows if r['phase'] == 'smoke']
    lines += ['## 规则引导GNN验证', '', f'验证完成{len(validation)}/96条；冒烟{len(smoke)}/8条，不进入统计。',
              '三种GNN组合均为旧模型贪心、旧模型采样、新模型固定模式贪心，各最多3次初始B32，不增加采样。'
              'CURRENT使用学习父选择和原logits；MRV仅改父选择；RULE_GUIDED使用MRV和规则/GNN名次融合。',
              '规则归属为软先验，所有合法动作保留。按GNN与规则名次和排序，同名次依次按规则成本、原logits和原候选顺序打破。'
              '贪心取第一名；原有采样槽使用负融合名次作为logits，保留原温度1和固定种子，不新增融合权重。',
              '四方法均使用启用安全剪枝的同一MODE_NEW、B32、CONFIG_05和60秒总预算；模型读取、构造、初始B32及搜索均计时。'
              '验证Excel读取在计时外。失败不回退、不剔除，Cmax只统计有效结果。', '',
              '|方法|可行/总次数|初始几何中位数均值|最终几何中位数均值|初始化平均秒|搜索平均秒|B32平均次数|B32平均秒|平均迭代|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    init_pairs = [r for r in rows if r['phase'] == 'initial_equivalence']
    if init_pairs:
        lines[-2:-2] = [f"初始化独立等价核查：{sum(r['status']=='EQUIVALENT' for r in init_pairs)}/{len(init_pairs)}组；"
                        '覆盖四个数量段和四种初始化，关闭墙钟截止，比较完整方案、方向和Cmax，B32次数不增加。', '']
    for method in display_methods:
        arm = [r for r in all_validation if r['method'] == method]
        if not arm:
            continue
        def avg(field):
            return statistics.fmean(float(r[field]) for r in arm)
        def mean_medians(field):
            values = defaultdict(list)
            for r in arm:
                if r[field] not in ('', None):
                    values[r['instance']].append(float(r[field]))
            return statistics.fmean(statistics.median(v) for v in values.values()) if values else None
        initial, final = mean_medians('initial_cmax'), mean_medians('final_cmax')
        lines.append(f"|{method}|{sum(bool(r['final_cmax']) for r in arm)}/{len(arm)}|"
                     f"{f'{initial:.3f}' if initial is not None else '—'}|{f'{final:.3f}' if final is not None else '—'}|"
                     + '|'.join(f'{avg(f):.3f}' for f in ('initialization_seconds','search_seconds','b32_calls','b32_seconds','iterations')) + '|')
    lines += ['', '先取同一几何三种子的有效Cmax中位数，再按8个几何比较；部分种子失败时样本量可能不同，'
              '条件Cmax不能抵消可行率下降。改善=(对照−规则引导)/对照，正值更好。', '',
              '|阶段|对照|规则引导胜/平/负|中位改善/%|', '|---|---|---|---:|']
    for field in ('initial_cmax', 'final_cmax'):
        for comparator in METHODS[:3]:
            counts, change = geometric_pair(validation, field, comparator)
            lines.append(f"|{field}|{comparator}|{'/'.join(map(str,counts))}|{f'{change:.3f}' if change is not None else '—'}|")
    lines += ['', '### 逐几何结果', '', '|实例|方法|有效/3|初始中位Cmax|最终中位Cmax|', '|---|---|---:|---:|---:|']
    for instance in sorted({r['instance'] for r in validation}):
        for method in display_methods:
            arm = [r for r in all_validation if r['instance'] == instance and r['method'] == method]
            initial = [float(r['initial_cmax']) for r in arm if r['initial_cmax']]
            final = [float(r['final_cmax']) for r in arm if r['final_cmax']]
            lines.append(f"|{instance}|{method}|{len(final)}/3|{f'{statistics.median(initial):.3f}' if initial else '—'}|"
                         f"{f'{statistics.median(final):.3f}' if final else '—'}|")
    errors = [r for r in rows if r['payload'].get('error')]
    for error in errors:
        lines += ['', f"异常：{error['phase']} / {error['instance']} / {error['method']}",
                  '```text', error['payload']['error'], '```']
    plans = [r for r in rows if r['phase'] == 'teacher_plan']
    teachers = [r for r in rows if r['phase'] in ('teacher','teacher_fallback')]
    training = [r for r in rows if r['phase'] == 'training']
    if plans:
        plan = plans[-1]['payload']
        complete = {r['instance'] for r in teachers if r['payload'].get('states')}
        lines += ['', '## 扩展训练', '',
                  '现有权重验证中，规则引导对CURRENT/MRV初始均8胜，最终分别7胜1负、8胜；'
                  '24/24可行，高于RULE的21/24，因此进入一次扩展训练。相对RULE仍不能宣称Cmax整体优势。',
                  f"原训练几何保留{plan['retained_geometries']}个、去重教师{plan['retained_teachers']}个；"
                  f"静态新增PPO训练几何{plan['additional_geometries']}个，与原几何重复数为{plan['duplicate_geometries']}。",
                  f'新增几何取得可执行教师：{len(complete)}/40。RULE和MODE_NEW共享一次真实初始化；'
                  '缺少教师时使用原WAG/HGA，不改变对照算法。只使用已记录的B32 FEASIBLE方案，'
                  '离线重放动作时不再B32或独立认证；不能重放的教师保留失败原因，不伪造动作标签。', '']
    for row in training:
        info = row['payload']; out = info['training']
        checkpoint = shared.read_neural_checkpoint(TRAINED_CHECKPOINT)
        replays = checkpoint['refinement_candidate']['refinement']['replays']
        supervised = {r['instance'] for r in replays if r['replay_status'] == 'REPLAYED'}
        new_supervised = {name for name in supervised if name.startswith('TRAIN_')}
        missing_supervised = sorted({r['instance'] for r in replays if r['instance'].startswith('TRAIN_')} - supervised)
        lines += [f"同一ModeJointInsertionGNN结构，加载原平衡模型继续训练{out['epochs']}轮；"
                  '沿用AdamW lr=0.001、batch=16、clip=1、seed=20261104和原负载辅助损失，无新层、无融合权重。'
                  '输入标准化只拟合训练轨迹，学习模式头不参与本轮训练/推理。',
                  f"完整教师池覆盖{info['coverage']['selected_geometries']}个几何；实际提供可重放监督的几何"
                  f"{len(supervised)}个（新增PPO {len(new_supervised)}个），形成{out['examples']}个训练样本；"
                  f"完整教师动作重放成功{info['replayed_teachers']}、失败{info['failed_replays']}。"
                  f"训练耗时{out['runtime_s']:.3f}s；新权重单独保存models/gnn_rule_guided_constructor_05.pt，原权重未写入。", '']
        lines += ['新增训练几何中未成功重放的为：' + '、'.join(missing_supervised) + '。'
                  '其完整教师仍保存，失败为既有128节点/2秒动作重放预算耗尽，未将这些几何伪计为已提供监督。', '']
    if retrained:
        lines += ['### 扩展训练后的同验证集复测', '',
                  f'完成{len(retrained)}/24条，复用同8个验证几何、同3个种子、同60秒总预算和MODE_NEW；'
                  '原96条作为对照，不重跑，holdout仍未打开。', '',
                  '|阶段|对照|新权重胜/平/负|中位改善/%|', '|---|---|---|---:|']
        for field in ('initial_cmax','final_cmax'):
            for comparator in METHODS:
                counts, change = geometric_pair(validation + retrained, field, comparator, 'RULE_GUIDED_GNN_RETRAINED')
                lines.append(f"|{field}|{comparator}|{'/'.join(map(str,counts))}|{f'{change:.3f}' if change is not None else '—'}|")
        lines += ['', f"新权重可行：{sum(bool(r['final_cmax']) for r in retrained)}/{len(retrained)}。", '']
    if len(validation) == 96 and len(retrained) == 24:
        lines += ['## 失败分析与下一步决定', '',
                  '1. **保留B32安全剪枝。** 固定工作量结果与接受轨迹一致，60秒比较增加迭代并取得9胜3平。'
                  '恢复策略、合法候选和Cmax目标均保留；测量不支持为模板生成另加缓存。',
                  '2. **规则融合优于当前GNN，但仍没有建立对RULE的整体Cmax优势。** 现有权重融合对CURRENT'
                  '初始8胜、最终7胜1负，中位改善24.772%/2.703%；对RULE的7个有效几何配对，'
                  '初始0胜7负、最终2胜1平4负，最终中位改善−0.362%。'
                  '融合24/24可行，RULE为21/24，故成功样本的总体均值不能直接横比。',
                  '3. **单独MRV不够。** 其N31–50第二个几何和较大几何终解明显较弱；'
                  'N71–90首个几何的一次初始化中，三个候选均被B32返回DEADLOCK。'
                  '规则融合在相同合法池内改善了排名，但P+MOVE排序与网络排名都不能保证更低真实协调Cmax。',
                  '4. **初始化失败来自有界候选菜单耗尽。** RULE在N31–50首个几何三种子共24次初始B32评价均返回DEADLOCK；'
                  '未取得任何可行初解时没有启用下界剪枝。该现象不等于证明几何不可行，其他GNN方法在同几何可行。',
                  '5. **本次扩展训练新增收益不足，停止继续加训。** 相对未扩训融合模型，初始为0胜7平1负、中位改善0%；'
                  '最终4胜2平2负、中位改善仅0.023%。7个几何初始中位Cmax完全相同，'
                  '不能把不同运行阶段的迭代速度差异全部归因于训练。对RULE最终仍为3胜4负、中位改善−0.316%，'
                  '尚不支持以新权重替换主方法。原权重与融合结果保留，新权重保存为本次研究证据，不继续堆叠模块。',
                  '主方法保留RULE初始化的MODE_NEW并使用已证明等价的B32安全剪枝；规则融合保留为开发候选。'
                  '8个保留holdout维持未打开状态，本报告仅为既有验证集上的开发证据。', '']
    checks = [r for r in rows if r['phase'] == 'verification']
    if checks:
        lines += ['## 软件和原始记录验证', '',
                  '以下60秒时间阈值核查针对96+24条验证运行；固定迭代等价测试有意关闭墙钟截止。', '']
        for check in checks:
            info = check['payload']
            lines += [info['summary'], '', '```powershell', info['command'], '```', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('profile', 'profile_detail', 'equivalence', 'time', 'data', 'smoke',
                                          'initial_equivalence', 'validation', 'teachers', 'train', 'retrained_validation', 'report'), default='profile')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'profile':
        run_profile()
    elif args.phase == 'profile_detail':
        run_profile_detail()
    elif args.phase in ('equivalence', 'time'):
        run_benchmark(args.phase, args.workers)
    elif args.phase == 'data':
        build_dataset()
    elif args.phase in ('smoke', 'validation', 'retrained_validation'):
        run_validation(args.phase, args.workers)
    elif args.phase == 'initial_equivalence':
        run_initial_equivalence(args.workers)
    elif args.phase == 'teachers':
        run_teachers(args.workers)
    elif args.phase == 'train':
        run_training()
    report()


if __name__ == '__main__':
    main()
