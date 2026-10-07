"""Phase4-0 local V2 oracle audit. No training, validation or test solvers.

Run with the specified venv from DRL. JSON holds the registered design and
contexts; SQLite transactions make individual oracle labels resumable.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass, asdict, replace
from datetime import datetime
from enum import Enum
from functools import lru_cache
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mrta_data.phase3_split import workbook_descriptors, _normalized, assert_v2_solver_access_allowed
from mrta_data.ppo_instances import load_ppo_platform_instance, instance_geometry_hash, to_parent_welds
from mrta_reference.model import ScientificConfig, CanonicalSolution, ScheduleStatus
from mrta_reference.geometry import pattern_catalog_hash
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.provenance import SourceProvenance, REPOSITORY_ID, compute_source_tree_hash
from mrta_reference.scheduler import FormalReferenceEvaluator
from mrta_reference.certifier import certify_schedule
from mrta_reference.solution import canonicalize
from mrta_search.pipeline import (SearchConfig, run_sa_oi_alns_v2, generate_candidate_pool,
    decision_family, select_c2_by_family, select_c4_by_family, DirectionEvaluatedCandidate)
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.stats import SearchStats
from scripts.run_phase3z_v2_validation import method_configs

SPLIT = ROOT / 'data/manifests/PPO_V2_RANKER_SPLIT_V1.json'
PROTOCOL = ROOT / 'data/manifests/PHASE4_0_ORACLE_RECALL_PROTOCOL_V1.json'
CONTEXTS = ROOT / 'data/development/phase4_0_state_contexts_v1.json'
LABELS = ROOT / 'data/development/phase4_0_candidate_oracle_labels_v1.sqlite'
RESULT = ROOT / 'data/development/phase4_0_candidate_pool_oracle_recall_v1.json'
REPORT = ROOT / 'docs/PHASE4_0_V2_CANDIDATE_POOL_ORACLE_RECALL_20261007.md'
ROLES = ROOT / 'data/manifests/PPO_V2_DATA_ROLES_V1.json'
DATASET = ROOT / 'data/manifests/PPO_DATASET_MANIFEST_V1.json'
PPO = ROOT.parent / 'ppo'
SEEDS = (20261031, 20261101)
TIERS = (('SMALL', 20, 30), ('MEDIUM', 50, 60), ('LARGE', 80, 90))
STAGES = (('EARLY', 5.0), ('MID', 30.0), ('LATE', 60.0))
CONFIG = ScientificConfig()
SEARCH = method_configs()['SA_OI_ALNS_V2']
FAMILIES = ('STRUCTURAL', 'TARGET_WHOLE', 'TARGET_Y', 'TARGET_X')


def now():
    return datetime.now().astimezone().isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    if path not in (SPLIT, PROTOCOL, CONTEXTS, RESULT):
        raise ValueError('Unapproved output path')
    path.write_text(dumps(value) + '\n', encoding='utf-8')


def encode(value, parents=None):
    """Explicit typed JSON, with shared instance geometry instead of pickle."""
    if isinstance(value, Enum):
        return {'enum': value.__class__.__module__ + ':' + value.__class__.__name__, 'value': value.value}
    if is_dataclass(value):
        return {'type': value.__class__.__module__ + ':' + value.__class__.__name__,
                'fields': {f.name: ({'shared_parents': True} if isinstance(value, CanonicalSolution)
                           and f.name == 'parents' and parents is not None and value.parents == parents
                           else encode(getattr(value, f.name), parents)) for f in fields(value)}}
    if isinstance(value, tuple):
        return {'tuple': [encode(v, parents) for v in value]}
    if isinstance(value, list):
        return [encode(v, parents) for v in value]
    if isinstance(value, dict):
        return {'map': [[encode(k, parents), encode(v, parents)] for k, v in value.items()]}
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def decode(value, parents=None):
    if isinstance(value, list):
        return [decode(v, parents) for v in value]
    if not isinstance(value, dict):
        return value
    if 'shared_parents' in value:
        if parents is None:
            raise ValueError('Missing shared geometry')
        return parents
    if 'tuple' in value:
        return tuple(decode(v, parents) for v in value['tuple'])
    if 'map' in value:
        return {decode(k, parents): decode(v, parents) for k, v in value['map']}
    if 'nonfinite' in value:
        return float(value['nonfinite'])
    tag = value.get('type', value.get('enum'))
    module, name = tag.split(':')
    if module not in ('mrta_reference.model', 'mrta_reference.certifier',
                      'mrta_search.lns', 'mrta_search.neighborhood', 'mrta_search.pipeline',
                      'mrta_search.direction'):
        raise ValueError('Unsupported stored type: ' + tag)
    cls = getattr(importlib.import_module(module), name)
    return cls(value['value']) if 'enum' in value else cls(**{k: decode(v, parents) for k, v in value['fields'].items()})


def pack(value, parents=None):
    return zlib.compress(dumps(encode(value, parents)).encode('utf-8'), 6)


def unpack(value, parents=None):
    return decode(json.loads(zlib.decompress(value)), parents)


def provenance():
    return SourceProvenance(REPOSITORY_ID, 'LOCAL_DRL_PHASE4_0_USER_AUTHORIZED',
                            compute_source_tree_hash(ROOT), True, False)


def valid_entries():
    return [e for e in read(DATASET)['instances'] if e['validation_status'] == 'VALID'
            and e['duplicate_of'] is None and 10 <= e['actual_weld_count'] <= 90]


def farthest_order(keys, vectors, tie):
    remaining = set(keys)
    chosen = []
    while remaining:
        def score(k):
            distance = min(sum((a-b)**2 for a,b in zip(vectors[k], vectors[s])) for s in chosen) if chosen else sum(a*a for a in vectors[k])
            return (-distance, tie[k], k)
        pick = min(remaining, key=score)
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def prepare_split():
    if SPLIT.exists():
        print('Existing split retained', flush=True)
        return
    roles = read(ROLES)
    paths = sorted(w['relative_path'] for w in roles['workbooks'] if w['new_v2_role'] == 'V2_TRAIN_POOL')
    assert len(paths) == 31
    entries = valid_entries()
    descriptors = workbook_descriptors(read(DATASET))
    tie = {p: digest(sorted(e['instance_geometry_hash'] for e in entries if e['relative_path'] == p)) for p in paths}
    order = farthest_order(paths, _normalized(descriptors, paths), tie)
    dev = order[:8]
    selected = []
    counts = Counter()
    # Metadata-only instance selection; each tier first spreads over workbooks.
    geom_fields = ('actual_weld_count', 'total_weld_length_m', 'bbox_area_ratio',
                   'x_coverage_ratio', 'y_coverage_ratio', 'cross_y6_count',
                   'cross_x10_count', 'upper_lower_length_imbalance', 'left_right_length_imbalance')
    for tier, lo, hi in TIERS:
        pool = [e for e in entries if e['relative_path'] in dev and lo <= e['actual_weld_count'] <= hi]
        vectors = {}
        means = [statistics.fmean(e[f] for e in pool) for f in geom_fields]
        scales = [statistics.pstdev(e[f] for e in pool) or 1.0 for f in geom_fields]
        index = {e['instance_id']: e for e in pool}
        for k,e in index.items():
            vectors[k] = [(e[f]-m)/s for f,m,s in zip(geom_fields, means, scales)]
        ordered = farthest_order(list(index), vectors, {k:e['instance_geometry_hash'] for k,e in index.items()})
        tier_used = set()
        for _ in range(4):
            options = [k for k in ordered if k not in {s['instance_id'] for s in selected} and counts[index[k]['relative_path']] < 2]
            diverse = [k for k in options if index[k]['relative_path'] not in tier_used]
            key = (diverse or options)[0]
            e = index[key]
            selected.append({**e, 'tier':tier, 'N':e['actual_weld_count'], 'ranker_role':'RANKER_DEV'})
            counts[e['relative_path']] += 1
            tier_used.add(e['relative_path'])
        assert len(tier_used) >= 2
    result = {'id':'PPO_V2_RANKER_SPLIT_V1', 'created_at':now(),
        'source_roles_hash':roles['v2_data_roles_hash'],
        'policy':'Whole-workbook z-score geometry farthest-first; max-radius first; ties workbook geometry hash then path. First 8 DEV, remaining 23 TRAIN. Instance tier z-score farthest-first, prefer distinct workbooks, max 2 per workbook.',
        'counts':{'RANKER_TRAIN':23,'RANKER_DEV':8},
        'descriptor_fields':list(next(iter(descriptors.values())).keys()),
        'workbooks':[{'relative_path':p,'parent_role':'V2_TRAIN_POOL','ranker_role':'RANKER_DEV' if p in dev else 'RANKER_TRAIN',
                      'workbook_geometry_hash':tie[p], 'descriptor':descriptors[p]} for p in paths],
        'farthest_first_order':order, 'audit_instances':selected}
    result['split_hash'] = digest(result)
    write(SPLIT, result)
    print(dumps({'split':'FROZEN','roles':result['counts'],'instances':[(e['tier'],e['N'],e['instance_id']) for e in selected]}),flush=True)


def load_parents(entry, *, debug=False):
    role = 'V2_MODEL_DEVELOPMENT_CONSUMED' if debug else 'V2_TRAIN_POOL'
    assert_v2_solver_access_allowed(read(ROLES), [entry['relative_path']], allowed_roles=(role,))
    if not debug:
        assigned = {e['relative_path']:e['ranker_role'] for e in read(SPLIT)['workbooks']}
        if assigned.get(entry['relative_path']) != 'RANKER_DEV':
            raise ValueError('Only RANKER_DEV may enter the oracle solver')
    instance = load_ppo_platform_instance(PPO/entry['relative_path'], entry['sheet_name'], ppo_root=PPO, instance_id=entry['instance_id'])
    if instance.raw_file_sha256 != entry['raw_file_sha256'] or instance_geometry_hash(instance) != entry['instance_geometry_hash']:
        raise ValueError('Workbook/geometry differs from registered input')
    return to_parent_welds(instance)


class Capture:
    """Retain immutable objects in memory. JSON encoding is outside the native clock."""
    def __init__(self, keep_all=False):
        self.latest = None
        self.snapshots = {}
        self.all = [] if keep_all else None

    def __call__(self, event, value):
        if event == 'boundary':
            value = dict(value)
            value['adaptive'] = deepcopy(value['adaptive'])
            value['attempts'] = []
            self.latest = value
            if self.all is not None:
                self.all.append(value)
            for stage, target in STAGES:
                if stage not in self.snapshots or value['elapsed'] <= target:
                    self.snapshots[stage] = value
        elif self.latest is not None:
            if event in ('atomic_attempt', 'lns_attempt'):
                self.latest['attempts'].append((event,value))
            else:
                self.latest[event] = value


def identity(candidate):
    return dumps(encode(candidate.candidate_identity))


def replay(context, *, seed):
    stats = SearchStats(FORMAL_SCOPE_V2.scope_id, seed)
    stats.iterations = context['iteration']
    adaptive = deepcopy(context['adaptive'])
    trace = []
    def observer(event, value):
        trace.append((event, value))
    generation_seed = seed + context['iteration'] * 65537
    screened, first, rng = generate_candidate_pool(context['solution'], context['directions'], CONFIG, SEARCH, stats,
        seed=generation_seed,current_schedule=context['schedule'],adaptive_state=adaptive,
        scope=FORMAL_SCOPE_V2,enable_x_split=True,observer=observer)
    if trace != context['attempts'] or tuple(first) != context.get('selection',((),()))[0]:
        raise ValueError('M64 actual attempt or candidate replay mismatch')
    c2 = select_c2_by_family(first, SEARCH.kdp, seed=seed, iteration=stats.iterations)
    if tuple(c2) != context['selection'][1]:
        raise ValueError('M64 C2 replay mismatch')
    ranks = {identity(c):i for i,c in enumerate(sorted(first,key=lambda c:c.cheap_score))}
    c3 = tuple(DirectionEvaluatedCandidate(c,ranks[identity(c)],optimize_directions_with_initial_feasibility(c.solution,CONFIG)) for c in c2)
    c4 = select_c4_by_family([c for c in c3 if c.direction.total_empty_travel is not None],SEARCH.kref,seed=seed,iteration=stats.iterations)
    observed_c3, observed_c4, _ = context['iteration_evaluation']
    if c3 != observed_c3 or tuple(identity(c.screened) for c in c4) != tuple(identity(c.direction_candidate.screened) for c in observed_c4):
        raise ValueError('M64 C3/C4 replay mismatch')
    complete = list(first)
    seen = {context['solution'].canonical_hash, *(c.solution.canonical_hash for c in first)}
    chunks = [(64,len(complete))]
    for atomic_offset,lns_offset,a,l,total in ((48,16,48,16,128),(96,32,96,32,256)):
        _, extra, rng = generate_candidate_pool(context['solution'],context['directions'],CONFIG,SEARCH,stats,
            seed=generation_seed,current_schedule=context['schedule'],adaptive_state=adaptive,
            scope=FORMAL_SCOPE_V2,enable_x_split=True,atomic_budget=a,lns_budget=l,
            atomic_offset=atomic_offset,lns_offset=lns_offset,lns_rng=rng,seen_solutions=seen,observer=observer)
        complete.extend(extra)
        chunks.append((total,len(complete)))
    if len(trace) != 256:
        raise ValueError('Attempt count differs from 256')
    return complete, trace, chunks, c2, c3, c4


def prepare_protocol():
    if PROTOCOL.exists():
        print('Existing preregistration retained', flush=True)
        return
    if not SPLIT.exists():
        raise ValueError('Prepare split first')
    p={'id':'PHASE4_0_ORACLE_RECALL_PROTOCOL_V1','registered_at':now(),
       'scope_hash':FORMAL_SCOPE_V2.scope_hash,'source':asdict(provenance()),
       'source_provenance_policy':'User explicitly requires execution in local DRL without standalone Git. commit_verified=False, dirty=True, formal_result=False; no clean-Git claim.',
       'split_hash':read(SPLIT)['split_hash'],'instances':read(SPLIT)['audit_instances'],
       'seeds':list(SEEDS),'search_config':asdict(SEARCH),'scientific_config':asdict(CONFIG),
       'stages':dict(STAGES),'snapshot_policy':'Last completed iteration boundary <= target, current accepted state. Certified initialization fallback only if no earlier completed search boundary. Preserve all stages even when identical. Missing native following M64 stream => execution FAIL; never fabricate it.',
       'shadow_stream':[[48,16],[48,16],[96,32]],'prefixes':[64,128,256],
       'continuation':'Atomic global attempt offsets, per-move ordinal continuation, same LNS RNG and adaptive selection context; no rejected/duplicate/identity refill.',
       'oracle':'Every unique valid M256 candidate: canonical V2 legality, production base direction DP, one B32 reference if direction-feasible, independent certification. No direction refinement. C4 base and POST_C4_DIRECTION_GAIN separately.',
       'statuses':['RAW_REJECTED','CONSTRUCTION_REJECTED','IDENTITY','DUPLICATE','DIRECTION_INFEASIBLE','FEASIBLE_CERTIFIED','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE'],
       'thresholds':{'epsilon_scale':1e-9,'opportunities_total':18,'opportunities_per_tier':4,
                     'generation_median':0.8,'generation_fraction_at_075':2/3,'generation_zero_max':0.25,
                     'ranking_median':0.8,'ranking_zero_max':0.25},
       'aggregation':'Equal state weighting; generation ratios on I256>epsilon, C2/C4 on I64>epsilon. NULL minima if no feasible; no-opportunity ratios NULL. Tier/stage/family auxiliary.',
       'decision_order':['EXECUTION_FAILURE','INSUFFICIENT_EVIDENCE','CANDIDATE_GENERATION','CANDIDATE_RANKING','NO_MATERIAL_POOL_OR_RANKING_GAP'],
       'ranking_stage':'C2 low when median C2 recall <.8 or zero capture >.25; C4 low under same criterion. Both -> C2_AND_C4. C4 recall denominator is I64.',
       'pool_sensitivity':'Generation FAIL and median generation recall128 >=.8',
       'mlp_authorized':'Only execution PASS + opportunity sufficient + generation PASS + ranking gap + zero numeric/certifier mismatch + isolation.',
       'label_key':['protocol_hash','state_context_id','complete_candidate_identity','scope_hash','source_tree_hash'],
       'debug_role':'V2_MODEL_DEVELOPMENT_CONSUMED','formal_audit_role':'RANKER_DEV',
       'regression_before_plan':{'passed':333},'regression_after_plan':{'passed':333}}
    p['protocol_hash']=digest(p)
    write(PROTOCOL,p)
    print(dumps({'protocol':'FROZEN','source':p['source']}),flush=True)


def verify_protocol():
    p=read(PROTOCOL)
    split=read(SPLIT)
    if digest({k:v for k,v in p.items() if k!='protocol_hash'})!=p['protocol_hash']:
        raise ValueError('Protocol content differs from registered identity')
    if digest({k:v for k,v in split.items() if k!='split_hash'})!=split['split_hash']:
        raise ValueError('Split content differs from registered identity')
    if split['source_roles_hash'] != read(ROLES)['v2_data_roles_hash']:
        raise ValueError('Historical role overlay changed')
    if p['source']['source_tree_hash'] != compute_source_tree_hash(ROOT):
        raise ValueError('Measured implementation differs from protocol source')
    if p['scope_hash'] != FORMAL_SCOPE_V2.scope_hash or p['split_hash'] != read(SPLIT)['split_hash']:
        raise ValueError('Protocol scope/split mismatch')
    if p['scientific_config'] != json.loads(dumps(asdict(CONFIG))):
        raise ValueError('Scientific configuration differs from preregistration')
    if p['search_config'] != json.loads(dumps(asdict(SEARCH))):
        raise ValueError('Production search configuration mismatch')
    return p


def debug():
    roles={e['relative_path']:e['new_v2_role'] for e in read(ROLES)['workbooks']}
    entries=[e for e in valid_entries() if roles[e['relative_path']]=='V2_MODEL_DEVELOPMENT_CONSUMED' and 20<=e['actual_weld_count']<=30]
    entry=min(entries,key=lambda e:e['instance_geometry_hash'])
    parents=load_parents(entry,debug=True)
    cfg=replace(SEARCH,max_iterations=2,time_limit=None)
    cap=Capture(keep_all=True)
    a=run_sa_oi_alns_v2(parents,CONFIG,cfg,seed=SEEDS[0],source_provenance=provenance())
    b=run_sa_oi_alns_v2(parents,CONFIG,cfg,seed=SEEDS[0],source_provenance=provenance(),observer=cap)
    assert a.best_solution==b.best_solution and a.best_directions==b.best_directions and a.best_metrics==b.best_metrics
    assert a.stats.proposal_trajectory==b.stats.proposal_trajectory
    assert a.stats.operator_sequence==b.stats.operator_sequence and a.stats.final_operator_weights==b.stats.final_operator_weights
    for context in cap.all[:-1]:
        replay(context,seed=SEEDS[0])
        assert unpack(pack(context,parents),parents)==context
    result={'instance_id':entry['instance_id'],'role':'V2_MODEL_DEVELOPMENT_CONSUMED',
            'iterations':2,'observer_on_off_equal':True,'actual_pool_replay_equal':True,
            'typed_storage_roundtrip_equal':True,'included_in_formal_aggregate':False}
    existing=read(RESULT) if RESULT.exists() else {}
    existing['debug']=result
    write(RESULT,existing)
    print(dumps(result),flush=True)


def trajectories():
    p=verify_protocol()
    data=read(CONTEXTS) if CONTEXTS.exists() else {'protocol_hash':p['protocol_hash'],'instances':{},'runs':[],'contexts':[]}
    if data['protocol_hash']!=p['protocol_hash']:
        raise ValueError('Context protocol mismatch')
    done={(r['instance_id'],r['seed']) for r in data['runs']}
    for ordinal,entry in enumerate(p['instances'],1):
        parents=load_parents(entry)
        data['instances'][entry['instance_id']]={'entry':entry,'parents':encode(parents),
            'pattern_catalog_hash':pattern_catalog_hash(parents,CONFIG,FORMAL_SCOPE_V2)}
        for seed in p['seeds']:
            if (entry['instance_id'],seed) in done:
                continue
            cap=Capture()
            cpu=time.process_time();wall=time.perf_counter()
            result=run_sa_oi_alns_v2(parents,CONFIG,SEARCH,seed=seed,source_provenance=provenance(),observer=cap)
            run={'instance_id':entry['instance_id'],'seed':seed,'tier':entry['tier'],
                 'status':result.status.value,'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu,
                 'native_runtime':result.runtime,'iterations':result.stats.iterations,'nref':result.stats.nref,
                 'scheduler_seconds':result.stats.reference_scheduler_time,'termination_reason':result.termination_reason,
                 'numeric_failures':result.stats.n_numeric_failure,
                 'certified':bool(result.final_certification and result.final_certification.certified),
                 'best_Cmax':None if result.best_metrics is None else result.best_metrics.cmax}
            for stage,target in STAGES:
                if stage not in cap.snapshots:
                    continue
                context=cap.snapshots[stage]
                context_id=f'{ordinal:02d}:{seed}:{stage}:{context["iteration"]}'
                data['contexts'].append({'state_context_id':context_id,'instance_id':entry['instance_id'],
                    'seed':seed,'tier':entry['tier'],'stage':stage,'target_seconds':target,
                    'elapsed':context['elapsed'],'iteration':context['iteration'],
                    'stage_source':'INITIAL' if context['iteration']==0 else 'SEARCH_ITERATION',
                    'initial_fallback_after_target':context['iteration']==0 and context['elapsed']>target,
                    'C_s':context['metrics'].cmax,'encoding':'zlib+base85 typed JSON; geometry in instances table',
                    'context':base64.b85encode(pack(context,parents)).decode('ascii')})
            data['runs'].append(run)
            write(CONTEXTS,data)
            print(dumps({'trajectory':len(data['runs']),'contexts':len(data['contexts']),**run}),flush=True)


def context_value(row,data):
    parents=decode(data['instances'][row['instance_id']]['parents'])
    return unpack(base64.b85decode(row['context']),parents)


def database(p):
    db=sqlite3.connect(LABELS)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS states(state_id TEXT PRIMARY KEY,metadata TEXT NOT NULL,replay_status TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS candidates(protocol_hash TEXT NOT NULL,state_id TEXT NOT NULL,
      candidate_id TEXT NOT NULL,scope_hash TEXT NOT NULL,source_hash TEXT NOT NULL,
      canonical_hash TEXT NOT NULL,family TEXT NOT NULL,first_prefix INTEGER NOT NULL,
      production_reference_evaluated INTEGER NOT NULL,payload BLOB NOT NULL,
      status TEXT,Cmax REAL,delta_Cmax REAL,audit_oracle_reference_evaluated INTEGER,
      direction_seconds REAL,reference_seconds REAL,cpu_seconds REAL,wall_seconds REAL,
      certifier_mismatch INTEGER,diagnostics TEXT,
      PRIMARY KEY(protocol_hash,state_id,candidate_id,scope_hash,source_hash),
      FOREIGN KEY(state_id) REFERENCES states(state_id));
    CREATE TABLE IF NOT EXISTS attempts(state_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
      source TEXT NOT NULL,status TEXT NOT NULL,family TEXT NOT NULL,candidate_id TEXT,
      details BLOB NOT NULL,PRIMARY KEY(state_id,ordinal),FOREIGN KEY(state_id) REFERENCES states(state_id));
    ''')
    existing=db.execute("SELECT value FROM metadata WHERE key='protocol_hash'").fetchone()
    if existing and existing[0]!=p['protocol_hash']:
        raise ValueError('Database belongs to another protocol')
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('protocol_hash',p['protocol_hash']))
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('source_hash',p['source']['source_tree_hash']))
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('scientific_config_hash',CONFIG.scientific_hash))
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('scope_hash',FORMAL_SCOPE_V2.scope_hash))
    conflicts=db.execute('SELECT count(*) FROM candidates WHERE protocol_hash<>? OR scope_hash<>? OR source_hash<>?',
                         (p['protocol_hash'],p['scope_hash'],p['source']['source_tree_hash'])).fetchone()[0]
    if conflicts:
        raise ValueError('Candidate labels have a different protocol/scope/source identity')
    db.commit()
    return db


def replay_all():
    p=verify_protocol();data=read(CONTEXTS);db=database(p)
    if data['protocol_hash'] != p['protocol_hash']:
        raise ValueError('Context protocol mismatch')
    for row in data['contexts']:
        state_id=row['state_context_id']
        if db.execute('SELECT 1 FROM states WHERE state_id=? AND replay_status=?',(state_id,'PASS')).fetchone():
            continue
        context=context_value(row,data)
        current_certification=certify_schedule(context['solution'],context['schedule'],CONFIG,scope=FORMAL_SCOPE_V2)
        if not current_certification.certified or context['schedule'].cmax != context['metrics'].cmax:
            raise ValueError('Frozen current state is not independently certified: '+state_id)
        started=time.perf_counter();cpu=time.process_time()
        complete,trace,chunks,c2,c3,c4=replay(context,seed=row['seed'])
        meta={k:v for k,v in row.items() if k!='context'}
        meta.update({'current_certified':True,'current_canonical_hash':context['solution'].canonical_hash,
                     'current_revision':context['solution'].revision,
                     'scientific_config_hash':CONFIG.scientific_hash,'scope_hash':FORMAL_SCOPE_V2.scope_hash,
                     'pattern_catalog_hash':data['instances'][row['instance_id']]['pattern_catalog_hash'],
                     'ranker_role':'RANKER_DEV'})
        meta.update({'C2':[identity(c) for c in c2],'C4':[identity(c.screened) for c in c4],
                     'cheap_order':[identity(c) for c in sorted(complete[:chunks[0][1]],key=lambda c:c.cheap_score)],
                     'C3_order':[identity(c.screened) for c in sorted([c for c in c3 if c.direction.total_empty_travel is not None],key=lambda c:c.rerank_key)],
                     'replay_wall_seconds':time.perf_counter()-started,'replay_cpu_seconds':time.process_time()-cpu})
        pre=context['iteration_evaluation'][2];post=context.get('post_c4')
        meta['POST_C4_DIRECTION_GAIN']=None if pre is None or post is None else pre.metrics.cmax-post.metrics.cmax
        meta['production_base_C4']=[{'candidate_id':identity(c.direction_candidate.screened),'status':c.schedule.status.value,
                                    'certified':bool(c.certification and c.certification.certified),'Cmax':None if c.metrics is None else c.metrics.cmax} for c in context['iteration_evaluation'][1]]
        lookup={c.solution.canonical_hash:c for c in complete}
        with db:
            db.execute('INSERT INTO states(state_id,metadata,replay_status) VALUES (?,?,?)',(state_id,dumps(meta),'PASS'))
            for i,c in enumerate(complete):
                prefix=next(n for n,end in chunks if i<end)
                db.execute('INSERT INTO candidates(protocol_hash,state_id,candidate_id,scope_hash,source_hash,canonical_hash,family,first_prefix,production_reference_evaluated,payload) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (p['protocol_hash'],state_id,stored_identity(identity(c)),p['scope_hash'],p['source']['source_tree_hash'],c.solution.canonical_hash,decision_family(c),prefix,int(identity(c) in meta['C4']),pack(c,context['solution'].parents)))
            seen={context['solution'].canonical_hash}
            for ordinal,(event,value) in enumerate(trace):
                if event=='atomic_attempt':
                    raw,status,reason,solution=value
                    family=(decision_family(raw.candidate) if raw.candidate is not None else
                            'TARGET_WHOLE' if raw.move_type.value=='SPLIT_DEACTIVATE' else
                            'UNRESOLVED_PATTERN' if raw.move_type.value in ('SPLIT_ACTIVATE','POINT_SWITCH') else 'STRUCTURAL')
                    source='ATOMIC'
                else:
                    _,pair,partial,repaired,status=value
                    solution=None if repaired.candidate is None else repaired.candidate.solution
                    family='STRUCTURAL';source='LNS_REPAIRED'
                candidate_id=None
                if solution is not None:
                    if solution.canonical_hash==context['solution'].canonical_hash:
                        status='IDENTITY'
                    elif solution.canonical_hash in seen:
                        status='DUPLICATE'
                    elif status=='VALID':
                        seen.add(solution.canonical_hash)
                        candidate_id=identity(lookup[solution.canonical_hash])
                db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(state_id,ordinal,source,status,family,stored_identity(candidate_id),pack((event,value),context['solution'].parents)))
        print(dumps({'replay':state_id,'status':'PASS','unique':chunks}),flush=True)
    db.close()


def evaluate_oracle_candidate(c, current_cmax, evaluator, production_base=()):
    wall=time.perf_counter();cpu=time.process_time()
    status=None;cmax=None;delta=None;ref=0;ref_time=0.0;dir_time=0.0;mismatch=0;diagnostics={}
    try:
        canonical=canonicalize(c.solution.parents,c.solution.patterns,c.solution.routes,CONFIG,revision=c.solution.revision,scope=FORMAL_SCOPE_V2)
        if canonical!=c.solution:
            raise ValueError('Candidate canonicalization changed payload')
        begin=time.perf_counter();direction=optimize_directions_with_initial_feasibility(c.solution,CONFIG);dir_time=time.perf_counter()-begin
        diagnostics['direction']=encode(direction)
        if direction.total_empty_travel is None:
            status='DIRECTION_INFEASIBLE'
        else:
            ref=1;begin=time.perf_counter()
            schedule=evaluator(c.solution,CONFIG,orientations={r:direction.directions[r] for r in range(4)})
            ref_time=time.perf_counter()-begin
            status=schedule.status.value
            diagnostics['schedule']=encode(schedule if not schedule.feasible else replace(schedule,operations=()),c.solution.parents)
            if schedule.status is ScheduleStatus.FEASIBLE:
                certification=certify_schedule(c.solution,schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                diagnostics['certification']=encode(certification)
                if certification.certified and schedule.cmax is not None and math.isfinite(schedule.cmax):
                    status='FEASIBLE_CERTIFIED';cmax=schedule.cmax;delta=current_cmax-cmax
                else:
                    status='NUMERIC_FAILURE';mismatch=1
                    diagnostics['schedule']=encode(schedule,c.solution.parents)
            observed=next((a for a in production_base if a['candidate_id']==identity(c)),None)
            if observed is not None and (observed['status']!=schedule.status.value or observed['Cmax']!=cmax):
                status='NUMERIC_FAILURE';mismatch=1;cmax=None;delta=None
                diagnostics['production_replay_mismatch']=observed
                diagnostics['schedule']=encode(schedule,c.solution.parents)
    except Exception:
        status='NUMERIC_FAILURE';cmax=None;delta=None
        diagnostics['exception']=traceback.format_exc()
    return (status,cmax,delta,ref,dir_time,ref_time,time.process_time()-cpu,
            time.perf_counter()-wall,mismatch,sqlite3.Binary(zlib.compress(dumps(diagnostics).encode('utf-8'))))


def oracle_worker(job):
    payload, parents, current_cmax, production_base = job
    candidate = unpack(payload, parents)
    label = evaluate_oracle_candidate(candidate, current_cmax,
                                      FormalReferenceEvaluator(FORMAL_SCOPE_V2), production_base)
    # sqlite3.Binary returns memoryview, which cannot cross a process boundary.
    return (*label[:-1], bytes(label[-1]))


def oracle():
    p=verify_protocol();data=read(CONTEXTS);db=database(p)
    if db.execute("SELECT count(*) FROM states WHERE replay_status='PASS'").fetchone()[0]!=72 or len(data['contexts'])!=72:
        raise ValueError('All 72 native pool replays must pass before oracle labels')
    rows={r['state_context_id']:r for r in data['contexts']}
    evaluator=FormalReferenceEvaluator(FORMAL_SCOPE_V2)
    total=db.execute('SELECT count(*) FROM candidates').fetchone()[0]
    finished=db.execute('SELECT count(*) FROM candidates WHERE status IS NOT NULL').fetchone()[0]
    # Offline labels have no search deadline and do not feed back into SA.
    # Three isolated processes use the same pure evaluator; the parent alone
    # writes transactions, in deterministic registered candidate order.
    started = time.perf_counter()
    previous=db.execute("SELECT value FROM metadata WHERE key='offline_elapsed_wall_seconds'").fetchone()
    prior_elapsed=float(previous[0]) if previous else 0.0
    with db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   ('offline_execution_mode', '3 isolated processes; single SQLite writer'))
    with ProcessPoolExecutor(max_workers=3) as executor:
        for state_id,row in rows.items():
            context=context_value(row,data);parents=context['solution'].parents
            meta=json.loads(db.execute('SELECT metadata FROM states WHERE state_id=?',(state_id,)).fetchone()[0])
            pending=[(cid,expanded_stored_blob(db,state_id,'candidate_payloads',payload)) for cid,payload in db.execute('SELECT candidate_id,payload FROM candidates WHERE state_id=? AND status IS NULL ORDER BY first_prefix,candidate_id',(state_id,)).fetchall()]
            jobs=((payload,parents,context['metrics'].cmax,meta['production_base_C4']) for _,payload in pending)
            for (candidate_id,_),label in zip(pending,executor.map(oracle_worker,jobs,chunksize=1)):
                status=label[0]
                with db:
                    db.execute('UPDATE candidates SET status=?,Cmax=?,delta_Cmax=?,audit_oracle_reference_evaluated=?,direction_seconds=?,reference_seconds=?,cpu_seconds=?,wall_seconds=?,certifier_mismatch=?,diagnostics=? WHERE protocol_hash=? AND state_id=? AND candidate_id=? AND scope_hash=? AND source_hash=?',
                        (*label,p['protocol_hash'],state_id,candidate_id,p['scope_hash'],p['source']['source_tree_hash']))
                with db:
                    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                               ('offline_elapsed_wall_seconds',str(prior_elapsed+time.perf_counter()-started)))
                finished+=1
                if finished%100==0 or status=='NUMERIC_FAILURE':
                    print(dumps({'labeled':finished,'total':total,'state':state_id,'status':status}),flush=True)
            print(dumps({'state_complete':state_id,'labeled':finished,'total':total}),flush=True)
    with db:
        elapsed=prior_elapsed+time.perf_counter()-started
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('offline_elapsed_wall_seconds',str(elapsed)))
    db.close()


def stored_identity(value):
    if value is None or isinstance(value,(bytes,memoryview)):
        return None if value is None else bytes(value)
    encoded=zlib.compress(value.encode('utf-8'),6)
    if zlib.decompress(encoded).decode('utf-8')!=value:
        raise ValueError('Identity compression changed content')
    return encoded


def expanded_identity(value):
    return zlib.decompress(bytes(value)).decode('utf-8') if isinstance(value,(bytes,memoryview)) else value


def expanded_stored_blob(db,state_id,packet_column,value,packet_cache=None):
    if value is None:
        return None
    if isinstance(value,str) and value.isdecimal():
        value=int(value)
    if not isinstance(value,int):
        return bytes(value)
    if packet_column not in ('attempt_trace','candidate_payloads','oracle_diagnostics'):
        raise ValueError('Unsupported shared packet column')
    key=(state_id,packet_column)
    if packet_cache is not None and key in packet_cache:
        entries=packet_cache[key]
    else:
        packet=db.execute('SELECT '+packet_column+' FROM states WHERE state_id=?',(state_id,)).fetchone()[0]
        entries=json.loads(zlib.decompress(packet))
        if packet_cache is not None:
            packet_cache[key]=entries
    encoded=entries[value]
    return zlib.compress(dumps(encoded).encode('utf-8'),6)


def read_attempt_detail(db,state_id,ordinal,parents=None,*,packet_cache=None):
    value=db.execute('SELECT details FROM attempts WHERE state_id=? AND ordinal=?',(state_id,ordinal)).fetchone()[0]
    return unpack(expanded_stored_blob(db,state_id,'attempt_trace',value,packet_cache),parents)


def read_candidate_payload(db,state_id,candidate_id,parents=None,*,packet_cache=None):
    value=db.execute("SELECT payload FROM candidates WHERE protocol_hash=(SELECT value FROM metadata WHERE key='protocol_hash') AND state_id=? AND candidate_id IN (?,?)",
                     (state_id,stored_identity(candidate_id),expanded_identity(candidate_id))).fetchone()[0]
    return unpack(expanded_stored_blob(db,state_id,'candidate_payloads',value,packet_cache),parents)


def read_candidate_diagnostic(db,state_id,candidate_id,*,packet_cache=None):
    value=db.execute("SELECT diagnostics FROM candidates WHERE protocol_hash=(SELECT value FROM metadata WHERE key='protocol_hash') AND state_id=? AND candidate_id IN (?,?)",
                     (state_id,stored_identity(candidate_id),expanded_identity(candidate_id))).fetchone()[0]
    blob=expanded_stored_blob(db,state_id,'oracle_diagnostics',value,packet_cache)
    return None if blob is None else json.loads(zlib.decompress(blob))


def compact_connection(db):
    """Lossless shared state packets, complete identities and ordinary primary keys."""
    db.execute('PRAGMA temp_store=MEMORY')
    db.create_function('expanded_identity',1,expanded_identity)
    db.create_function('stored_identity',1,stored_identity)
    columns={table:[r[1] for r in db.execute('PRAGMA table_info('+table+')')]
             for table in ('candidates','attempts')}
    packets={}
    @lru_cache(maxsize=6)
    def raw_packet(state_id,column):
        return [dumps(v).encode('utf-8') for v in json.loads(zlib.decompress(packets[state_id,column]))]
    def expanded_blob(state_id,column,value):
        if value is None:
            return None
        if isinstance(value,str) and value.isdecimal():
            value=int(value)
        return zlib.compress(raw_packet(state_id,column)[value],6) if isinstance(value,int) else bytes(value)
    with db:
        db.execute('CREATE TEMP TABLE original_candidates AS SELECT * FROM candidates')
        db.execute('CREATE TEMP TABLE original_attempts AS SELECT * FROM attempts')
        present=[r[1] for r in db.execute('PRAGMA table_info(states)')]
        for column in ('attempt_trace','candidate_payloads','oracle_diagnostics'):
            if column not in present:
                db.execute('ALTER TABLE states ADD COLUMN '+column+' BLOB')
        protocol_hash=db.execute("SELECT value FROM metadata WHERE key='protocol_hash'").fetchone()[0]
        for state_id in [r[0] for r in db.execute('SELECT state_id FROM states').fetchall()]:
            for table,column,packet_column,key in (('attempts','details','attempt_trace','rowid'),
                    ('candidates','payload','candidate_payloads','rowid'),
                    ('candidates','diagnostics','oracle_diagnostics','rowid')):
                packet=db.execute('SELECT '+packet_column+' FROM states WHERE state_id=?',(state_id,)).fetchone()[0]
                if packet is None:
                    where='protocol_hash=? AND state_id=?' if table=='candidates' else 'state_id=?'
                    parameters=(protocol_hash,state_id) if table=='candidates' else (state_id,)
                    records=db.execute('SELECT '+key+','+column+' FROM '+table+' WHERE '+where+' ORDER BY '+key,parameters).fetchall()
                    encoded=[None if value is None else json.loads(zlib.decompress(value)) for _,value in records]
                    packet=zlib.compress(dumps(encoded).encode('utf-8'),9)
                    restored=json.loads(zlib.decompress(packet))
                    if any(value is not None and zlib.compress(dumps(v).encode('utf-8'),6)!=bytes(value) for v,(_,value) in zip(restored,records)):
                        raise ValueError('Shared packet changed a stored blob')
                    db.execute('UPDATE states SET '+packet_column+'=? WHERE state_id=?',(packet,state_id))
                    db.executemany('UPDATE '+table+' SET '+column+'=? WHERE rowid=?',
                        [(None if value is None else ordinal,row_key) for ordinal,(row_key,value) in enumerate(records)])
                packets[state_id,packet_column]=packet
        db.create_function('expanded_blob',3,expanded_blob)
        db.execute('UPDATE candidates SET candidate_id=stored_identity(candidate_id)')
        db.execute('UPDATE attempts SET candidate_id=stored_identity(candidate_id) WHERE candidate_id IS NOT NULL')
        for table in ('candidates','attempts'):
            expressions=[]
            for column in columns[table]:
                packet_column=({'details':'attempt_trace'} if table=='attempts' else
                               {'payload':'candidate_payloads','diagnostics':'oracle_diagnostics'}).get(column)
                expressions.append('expanded_identity(candidate_id)' if column=='candidate_id' else
                    "expanded_blob(state_id,'"+packet_column+"',"+column+")" if packet_column else column)
            projection=','.join(expressions)
            old='SELECT '+projection+' FROM original_'+table
            new='SELECT '+projection+' FROM '+table
            if db.execute('SELECT 1 FROM ('+old+' EXCEPT '+new+') LIMIT 1').fetchone() or db.execute('SELECT 1 FROM ('+new+' EXCEPT '+old+') LIMIT 1').fetchone():
                raise ValueError('Logical rows changed during compaction: '+table)
        db.execute('DROP TABLE original_candidates')
        db.execute('DROP TABLE original_attempts')
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   ('identity_encoding','zlib UTF-8 complete identity; legacy TEXT readable'))
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   ('blob_encoding','state-level zlib typed JSON packets; integer row references; legacy per-row BLOB readable'))
    return True


def compact_storage():
    p=verify_protocol();db=database(p)
    if db.execute('SELECT count(*) FROM candidates WHERE status IS NULL').fetchone()[0]:
        raise ValueError('Finish oracle labels before storage compaction')
    before=LABELS.stat().st_size
    wall_started=time.perf_counter();cpu_started=time.process_time()
    compact_connection(db)
    db.execute('VACUUM')
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.execute('PRAGMA foreign_key_check').fetchall():
        raise ValueError('SQLite integrity/foreign-key check failed')
    after=LABELS.stat().st_size
    result=read(RESULT)
    result['storage_compaction']={'bytes_before':before,'bytes_after':after,
        'reduction_percent':100*(before-after)/before,
        'format':'compressed complete identity + shared state packets for candidates/attempts/diagnostics + SQLite VACUUM',
        'all_logical_rows_equal_after_expansion':True,
        'cpu_seconds':time.process_time()-cpu_started,'wall_seconds':time.perf_counter()-wall_started,
        'sqlite_integrity':'ok','foreign_keys':'ok'}
    write(RESULT,result);db.close()
    print(dumps(result['storage_compaction']),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare-split','debug','prepare','trajectories','replay','oracle','compact-storage','summarize'))
    args=parser.parse_args()
    {'prepare-split':prepare_split,'debug':debug,'prepare':prepare_protocol,'trajectories':trajectories,
     'replay':replay_all,'oracle':oracle,'compact-storage':compact_storage,'summarize':summarize}[args.command]()


def recall_metrics(current, candidates, c2_ids, c4_ids):
    epsilon=1e-9*max(1.0,abs(current))
    feasible=[c for c in candidates if c['status']=='FEASIBLE_CERTIFIED']
    pools={str(n):[c for c in feasible if c['first_prefix']<=n] for n in (64,128,256)}
    pools['C2']=[c for c in feasible if c['candidate_id'] in c2_ids]
    pools['C4']=[c for c in feasible if c['candidate_id'] in c4_ids]
    minima={k:min((c['Cmax'] for c in pool),default=None) for k,pool in pools.items()}
    improvements={k:0.0 if v is None else max(0.0,current-v) for k,v in minima.items()}
    if not improvements['256']+epsilon>=improvements['128']>=improvements['64']-epsilon:
        raise ValueError('Non-nested improvement minima')
    opportunity=improvements['256']>epsilon
    captured=improvements['64']>epsilon
    def ratio(n,d):
        if improvements[d]<=epsilon:
            return None
        value=improvements[n]/improvements[d]
        if not -1e-12<=value<=1+1e-12:
            raise ValueError('Recall outside [0,1]')
        return min(1.0,max(0.0,value))
    def hit(small,large):
        if minima[large] is None:
            return None
        return minima[small] is not None and minima[small]<=minima[large]+epsilon
    return {'C_s':current,'epsilon_C':epsilon,'C_star':minima,'I':improvements,
        'no_feasible':{k:v is None for k,v in minima.items()},'opportunity':opportunity,
        'generation_recall64':ratio('64','256'),'generation_recall128':ratio('128','256'),
        'c2_recall':ratio('C2','64'),'c4_recall':ratio('C4','64'),
        'c4_conditional_c2_recall':ratio('C4','C2'),
        'zero_capture64':opportunity and not captured,
        'zero_captureC2':captured and improvements['C2']<=epsilon,
        'zero_captureC4':captured and improvements['C4']<=epsilon,
        'top1_64_of256':hit('64','256'),'top1_128_of256':hit('128','256'),
        'top1_C2_of64':hit('C2','64'),'top1_C4_of64':hit('C4','64')}


def distribution(values):
    values=sorted(v for v in values if v is not None)
    if not values:
        return {'count':0,'mean':None,'median':None,'minimum':None,'maximum':None}
    return {'count':len(values),'mean':statistics.fmean(values),'median':statistics.median(values),
            'minimum':values[0],'maximum':values[-1]}


def aggregate(rows):
    opportunities=[r for r in rows if r['opportunity']]
    captured=[r for r in rows if r['I']['64']>r['epsilon_C']]
    result={'states':len(rows),'opportunities':len(opportunities),'M64_opportunities':len(captured),
        **{k:distribution([r[k] for r in rows]) for k in ('generation_recall64','generation_recall128','c2_recall','c4_recall')},
        'fraction_gen64_at_least_075':None if not opportunities else sum(r['generation_recall64']>=.75 for r in opportunities)/len(opportunities),
        'zero_capture64_fraction':None if not opportunities else sum(r['zero_capture64'] for r in opportunities)/len(opportunities),
        'zero_captureC2_fraction':None if not captured else sum(r['zero_captureC2'] for r in captured)/len(captured),
        'zero_captureC4_fraction':None if not captured else sum(r['zero_captureC4'] for r in captured)/len(captured)}
    for n in (64,128,256):
        result['unique_'+str(n)]=distribution([r.get('unique_counts',{}).get(str(n)) for r in rows])
        result['improvement_'+str(n)]=distribution([r['I'][str(n)] for r in rows])
    for key in ('top1_64_of256','top1_128_of256','top1_C2_of64','top1_C4_of64'):
        result[key]=distribution([r[key] for r in rows])
    return result


def decisions(rows, execution_pass):
    overall=aggregate(rows)
    sufficient=overall['opportunities']>=18 and all(sum(r['opportunity'] for r in rows if r['tier']==tier)>=4 for tier,_,_ in TIERS)
    generation=None
    gap=None
    sensitivity=None
    ranking_stage='NOT_EVALUABLE'
    bottleneck='NOT_EVALUABLE'
    next_phase='Resolve Phase4-0 execution failure'
    if execution_pass:
        if not sufficient:
            bottleneck='INSUFFICIENT_EVIDENCE'
            next_phase='Phase4-0 evidence extension: statically select more RANKER_DEV states or preregister additional seeds'
        else:
            generation=(overall['generation_recall64']['median']>=.8 and overall['fraction_gen64_at_least_075']>=2/3 and overall['zero_capture64_fraction']<=.25)
            sensitivity=not generation and overall['generation_recall128']['median']>=.8
            if not generation:
                bottleneck='CANDIDATE_GENERATION'
                next_phase='Phase4-0B Bounded Candidate Generation Scaling Audit'
            else:
                low2=overall['c2_recall']['median']<.8 or overall['zero_captureC2_fraction']>.25
                low4=overall['c4_recall']['median']<.8 or overall['zero_captureC4_fraction']>.25
                gap=low4
                ranking_stage=('C2_AND_C4' if low2 and low4 else 'C2_CHEAP_SCREEN' if low2 else 'C4_RERANK_REFERENCE_SHORTLIST' if low4 else 'NONE')
                bottleneck='CANDIDATE_RANKING' if gap else 'NO_MATERIAL_POOL_OR_RANKING_GAP'
                next_phase='Phase4-1 MLP Fixed-Pool Learned Screening' if gap else 'Audit SA acceptance / state trajectory / horizon'
    return {'PHASE4_0_EXECUTION_STATUS':'PASS' if execution_pass else 'FAIL',
        'AUTHORITATIVE_PLAN_V2_SYNC':'PASS','RANKER_SPLIT_STATUS':'FROZEN',
        'AUDIT_STATE_CONTEXTS':str(len(rows))+' / 72',
        'CANDIDATE_POOL_REPLAY_STATUS':'PASS' if len(rows)==72 else 'FAIL',
        'OPPORTUNITY_EVIDENCE_SUFFICIENT':'YES' if sufficient else 'NO',
        'GENERATION_COVERAGE_STATUS':'NOT_EVALUABLE' if generation is None else 'PASS' if generation else 'FAIL',
        'RANKING_GAP_STATUS':'NOT_EVALUABLE' if gap is None else 'SUPPORTED' if gap else 'NOT_SUPPORTED',
        'PRIMARY_RANKING_STAGE':ranking_stage,'PRIMARY_BOTTLENECK':bottleneck,
        'POOL_SIZE_SENSITIVITY':'NOT_EVALUABLE' if sensitivity is None else 'SUPPORTED' if sensitivity else 'NOT_SUPPORTED',
        'PHASE4_1_MLP_AUTHORIZED':'YES' if execution_pass and sufficient and generation and gap else 'NO',
        'TRADITIONAL_HEURISTIC_TUNING_STATUS':'CLOSED','V2_VALIDATION_STATUS':'CONSUMED',
        'ID_TEST_STATUS':'SEALED','NEXT_PHASE':next_phase}


def summarize():
    p=verify_protocol();data=read(CONTEXTS);db=database(p)
    db.row_factory=sqlite3.Row
    remaining=db.execute('SELECT count(*) FROM candidates WHERE status IS NULL').fetchone()[0]
    if remaining:
        raise ValueError(f'{remaining} candidates remain unlabeled')
    rows=[]
    for state in db.execute('SELECT * FROM states ORDER BY state_id').fetchall():
        meta=json.loads(state['metadata'])
        candidates=[dict(c) for c in db.execute('SELECT candidate_id,family,first_prefix,status,Cmax,delta_Cmax,production_reference_evaluated,audit_oracle_reference_evaluated,certifier_mismatch FROM candidates WHERE state_id=?',(state['state_id'],))]
        for candidate in candidates:
            candidate['candidate_id']=expanded_identity(candidate['candidate_id'])
        attempts=[dict(a) for a in db.execute('SELECT ordinal,status,family,candidate_id,source FROM attempts WHERE state_id=? ORDER BY ordinal',(state['state_id'],))]
        for attempt in attempts:
            attempt['candidate_id']=expanded_identity(attempt['candidate_id'])
        by_id={c['candidate_id']:c for c in candidates}
        metrics=recall_metrics(meta['C_s'],candidates,meta['C2'],meta['C4'])
        row={**meta,**metrics,'unique_counts':{},'counts':{},'families':{}}
        for n in (64,128,256):
            included=[c for c in candidates if c['first_prefix']<=n]
            prefix=attempts[:n]
            counts=Counter(by_id[a['candidate_id']]['status'] if a['candidate_id'] is not None else a['status'] for a in prefix)
            row['counts'][str(n)]={status:counts.get(status,0) for status in p['statuses']}
            if sum(row['counts'][str(n)].values())!=n:
                raise ValueError('Attempt status accounting differs from prefix budget')
            row['unique_counts'][str(n)]=len(included)
            family_rows={}
            for family in (*FAMILIES,'UNRESOLVED_PATTERN'):
                members=[c for c in included if c['family']==family]
                feasible=[c for c in members if c['status']=='FEASIBLE_CERTIFIED']
                best=min((c['Cmax'] for c in feasible),default=None)
                family_rows[family]={'attempts':sum(a['family']==family for a in prefix),'valid':len(members),
                    'direction_feasible':sum(c['audit_oracle_reference_evaluated'] or 0 for c in members),
                    'reference_feasible':len(feasible),'improving':sum(c['delta_Cmax']>metrics['epsilon_C'] for c in feasible),
                    'best_improvement':0.0 if best is None else max(0.0,meta['C_s']-best),
                    'oracle_best_ties':sum(abs(c['Cmax']-metrics['C_star'][str(n)])<=metrics['epsilon_C'] for c in feasible),
                    'oracle_best_not_production_evaluated':sum(abs(c['Cmax']-metrics['C_star'][str(n)])<=metrics['epsilon_C'] and not c['production_reference_evaluated'] for c in feasible),
                    'C2_selected':sum(c['candidate_id'] in meta['C2'] for c in members),
                    'production_reference_evaluated':sum(c['production_reference_evaluated'] for c in members)}
            row['families'][str(n)]=family_rows
        best64=metrics['C_star']['64']
        ties=[] if best64 is None else [c['candidate_id'] for c in candidates if c['first_prefix']==64 and c['status']=='FEASIBLE_CERTIFIED' and abs(c['Cmax']-best64)<=metrics['epsilon_C']]
        row['best64_cheap_ranks']=[meta['cheap_order'].index(c)+1 for c in ties if c in meta['cheap_order']]
        row['best64_C3_ranks']=[meta['C3_order'].index(c)+1 for c in ties if c in meta['C3_order']]
        row['best64_not_in_C3']=sum(c not in meta['C3_order'] for c in ties)
        rows.append(row)
    numeric=db.execute("SELECT count(*) FROM candidates WHERE status='NUMERIC_FAILURE'").fetchone()[0]
    mismatches=db.execute('SELECT coalesce(sum(certifier_mismatch),0) FROM candidates').fetchone()[0]
    trajectory_ok=len(data['runs'])==24 and all(r['certified'] and not r['numeric_failures'] for r in data['runs'])
    existing=read(RESULT)
    execution=(len(rows)==72 and trajectory_ok and numeric==0 and mismatches==0
               and existing.get('regression',{}).get('passed',0)>333 and existing.get('debug',{}).get('observer_on_off_equal',False))
    costs=dict(db.execute('SELECT count(*) as candidates,sum(audit_oracle_reference_evaluated) as reference_calls,sum(reference_seconds) as reference_seconds,sum(direction_seconds) as direction_seconds,sum(cpu_seconds) as cpu_seconds,sum(wall_seconds) as wall_seconds FROM candidates').fetchone())
    costs['candidate_evaluator_cpu_seconds']=costs.pop('cpu_seconds')
    costs['candidate_evaluator_wall_seconds']=costs.pop('wall_seconds')
    complete_cpu=db.execute("SELECT value FROM metadata WHERE key='offline_complete_process_cpu'").fetchone()
    costs['offline_complete_process_cpu']=None if complete_cpu is None else json.loads(complete_cpu[0])
    costs['offline_total_cpu_seconds']=None if complete_cpu is None else costs['offline_complete_process_cpu']['total_cpu_seconds']
    costs['offline_execution_mode']=db.execute("SELECT value FROM metadata WHERE key='offline_execution_mode'").fetchone()[0]
    costs['offline_elapsed_wall_seconds']=float(db.execute("SELECT value FROM metadata WHERE key='offline_elapsed_wall_seconds'").fetchone()[0])
    costs['replay_wall_seconds']=sum(r['replay_wall_seconds'] for r in rows)
    costs['replay_cpu_seconds']=sum(r['replay_cpu_seconds'] for r in rows)
    costs['trajectory_cpu_seconds']=sum(r['cpu_seconds'] for r in data['runs'])
    costs['trajectory_wall_seconds']=sum(r['wall_seconds'] for r in data['runs'])
    costs['trajectory_reference_calls']=sum(r['nref'] for r in data['runs'])
    result={**existing,'protocol_hash':p['protocol_hash'],'ranker_split_hash':p['split_hash'],
        'scope_hash':p['scope_hash'],'reference_scheduler_policy_id':FORMAL_SCOPE_V2.reference_scheduler_policy_id,
        'updated_at':now(),'source':p['source'],
        'status':decisions(rows,execution),'numeric_failures':numeric,'certifier_mismatches':mismatches,
        'costs':costs,'overall':aggregate(rows),'by_tier':{tier:aggregate([r for r in rows if r['tier']==tier]) for tier,_,_ in TIERS},
        'by_stage':{stage:aggregate([r for r in rows if r['stage']==stage]) for stage,_ in STAGES},
        'family_mean_by_state':{str(n):{f:{metric:statistics.fmean(r['families'][str(n)][f][metric] for r in rows) for metric in rows[0]['families'][str(n)][f]} for f in FAMILIES} for n in (64,128,256)},
        'opportunity_oracle_best_family_ties':{f:sum(r['families']['256'][f]['oracle_best_ties'] for r in rows if r['opportunity']) for f in FAMILIES},
        'candidate_status_counts':dict(db.execute('SELECT status,count(*) FROM candidates GROUP BY status').fetchall()),
        'attempt_status_counts':{str(n):{status:sum(r['counts'][str(n)][status] for r in rows) for status in p['statuses']} for n in (64,128,256)},
        'oracle_best_family_ties_by_tier_prefix':{tier:{str(n):{f:sum(r['families'][str(n)][f]['oracle_best_ties'] for r in rows if r['tier']==tier and r['opportunity']) for f in FAMILIES} for n in (64,128,256)} for tier,_,_ in TIERS},
        'initial_stage_fallbacks':sum(r['stage_source']=='INITIAL' for r in data['contexts']),
        'scientific_config_hash':CONFIG.scientific_hash,
        'post_c4_direction_gain':distribution([r['POST_C4_DIRECTION_GAIN'] for r in rows]),
        'c4_conditional_c2_recall_auxiliary':distribution([r['c4_conditional_c2_recall'] for r in rows]),'states':rows}
    write(RESULT,result)
    db.close()
    write_report(result)
    print(dumps(result['status']),flush=True)


def write_report(result):
    o=result['overall'];st=result['status']
    large=result['by_tier']['LARGE']
    large_gen=large['generation_recall64']['median']
    large_c4=large['c4_recall']['median']
    large_evidence=('opportunity不足，无法定位' if large['opportunities']<4 else
        '生成覆盖损失明显' if large_gen is not None and (large_gen<.8 or large['fraction_gen64_at_least_075']<2/3 or large['zero_capture64_fraction']>.25) else
        '更符合筛选排名损失' if large_c4 is not None and (large_c4<.8 or large['zero_captureC4_fraction']>.25) else '未见明显生成或排名损失')
    def pct(value):
        return 'NA' if value is None else f'{100*value:.2f}%'
    x=result['family_mean_by_state']['64']['TARGET_X']
    lines=['# Phase4-0 V2 Candidate-Pool Oracle Recall Audit', '',
        '本轮使用用户指定的本地 DRL 与 Python 环境。独立 Git 要求由用户明确撤销；source_commit 为本地标签，commit_verified=False、dirty=True，未声称正式 clean Git provenance。未上传 GitHub。', '',
        '固定生产 M64 / atomic48 + LNS16 / Kdp8 / Kref2 / Kref_total4 / B32。影子池按 48+16、48+16、96+32 的真实尝试顺序继续生成。坏候选、重复和 identity 均占预算，不补抽。Oracle 仅使用基础方向 DP；方向细化收益独立记录。', '',
        '## 状态', '', '```']
    lines += [k+' = '+str(v) for k,v in st.items()]
    lines += ['```','','## 16 个问题的测量回答','',
        '1. M64、M128、M256 严格嵌套，72 个实际轨迹池逐尝试、构造解、身份及 C2/C4 重放一致。',
        f"2. M64 有效 unique candidate：平均 {o['unique_64']['mean']:.3f}，中位 {o['unique_64']['median']:.3f}。",
        f"3. M256 有 {o['opportunities']} 个改善机会状态；M64 有 {o['M64_opportunities']} 个，相差 {o['opportunities']-o['M64_opportunities']} 个。",
        f"4. 在 {o['opportunities']} 个改善机会状态中，M64 捕获改善的中位比例为 {pct(o['generation_recall64']['median'])}；达到 75% 捕获率的状态占 {pct(o['fraction_gen64_at_least_075'])}，完全未捕获的占 {pct(o['zero_capture64_fraction'])}。",
        f"5. M128 捕获比例中位数为 {pct(o['generation_recall128']['median'])}，均值为 {pct(o['generation_recall128']['mean'])}；典型状态接近 M256，但尾部仍有损失，不能说所有状态已足够。按冻结规则，池大小敏感性 {st['POOL_SIZE_SENSITIVITY']}。",
        f"6. 在 {o['M64_opportunities']} 个 M64 有改善的状态中，C2 中位保留 {pct(o['c2_recall']['median'])}，中位损失 {pct(None if o['c2_recall']['median'] is None else 1-o['c2_recall']['median'])}；完全未捕获占 {pct(o['zero_captureC2_fraction'])}。",
        f"7. C4 最终中位保留 {pct(o['c4_recall']['median'])}，完全未捕获占 {pct(o['zero_captureC4_fraction'])}；分母始终为 M64 可利用改善 I64。",
        f"8. 机械判定主要瓶颈 {st['PRIMARY_BOTTLENECK']}，ranking stage {st['PRIMARY_RANKING_STAGE']}。全局 generation median 虽高，但达到 75% 捕获率的状态比例为 {pct(o['fraction_gen64_at_least_075'])}，未满足至少 2/3 的覆盖条件。冻结规则要求先通过 generation 才判 ranking；C2/C4 低保留率仍是测量事实，不能解释成不存在筛选损失。",
        '9. SMALL/MEDIUM/LARGE 的等状态权重分层结果见下表。',
        '10. EARLY/MID/LATE 结果见下表；相同解但不同阶段/上下文仍保留。',
        '11. M256 改善机会状态中的 oracle-best family ties：'+dumps(result['opportunity_oracle_best_family_ties'])+'；并列最优分别计数，不推成独占贡献。',
        f"12. X 在 M64 每状态平均尝试 {x['attempts']:.3f}、有效 {x['valid']:.3f}、生产 reference 评价 {x['production_reference_evaluated']:.3f}；X 达到 M64 oracle-best 但未被生产评价的候选平均 {x['oracle_best_not_production_evaluated']:.3f}。原始拒绝不能确定 pattern 的另记 UNRESOLVED_PATTERN；未评价的并列最优不自动等于改善损失。",
        f"13. LARGE：{large_evidence}。24 个状态中有 {large['opportunities']} 个 M256 改善机会；M64 捕获中位 {pct(large_gen)}、均值 {pct(large['generation_recall64']['mean'])}，达到 75% 捕获率的占 {pct(large['fraction_gen64_at_least_075'])}，完全未捕获占 {pct(large['zero_capture64_fraction'])}。M128 捕获中位 {pct(large['generation_recall128']['median'])}、均值 {pct(large['generation_recall128']['mean'])}。在 {large['M64_opportunities']} 个 M64 有改善的状态中，C2 中位保留 {pct(large['c2_recall']['median'])}，C4 中位保留 {pct(large_c4)}，两层都存在损失；C4 完全未捕获占 {pct(large['zero_captureC4_fraction'])}。所以不能将全局机械标签直接当作 LARGE 退化的成因。这只是开发状态上的机会损失证据。",
        f"14. MLP 授权 {st['PHASE4_1_MLP_AUTHORIZED']}；必须同时通过执行、机会数、生成覆盖和排名差距条件。",
        '15. 本轮未验证图表示相对固定候选特征的收益；即使 MLP 获准，也不能据此推进 GAT。',
        f"16. 下一阶段：{st['NEXT_PHASE']}。本轮不执行该阶段。", '',
        '## 分层结果', '', '| 分组 | 状态 | 改善机会 | median gen64 | median gen128 | median C2 | median C4 |', '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in {**result['by_tier'],**result['by_stage']}.items():
        values=[v[m]['median'] for m in ('generation_recall64','generation_recall128','c2_recall','c4_recall')]
        lines.append('| '+k+' | '+str(v['states'])+' | '+str(v['opportunities'])+' | '+' | '.join('NA' if x is None else f'{x:.6f}' for x in values)+' |')
    lines += ['', '## Family（每状态平均；M64）', '', '| family | 尝试 | 有效 | C2选择 | 生产reference | reference可行 | 改善 | oracle-best ties |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f,v in result['family_mean_by_state']['64'].items():
        lines.append('| '+f+' | '+' | '.join(f"{v[k]:.3f}" for k in ('attempts','valid','C2_selected','production_reference_evaluated','reference_feasible','improving','oracle_best_ties'))+' |')
    lines += ['', '## LARGE 扩池后的 oracle-best family', '',
        '在同一批 M256 改善机会状态上，以下统计各 prefix 的最优候选并列数量；扩池会替换原有最优，数量并不单调，也不是独立状态数。M256 中以 STRUCTURAL 和 WHOLE 为主，Y/X 没有达到扩池后的最优；不能据此修改 family quota。', '',
        '| prefix | STRUCTURAL | WHOLE | Y | X |', '|---|---:|---:|---:|---:|']
    for n, families in result['oracle_best_family_ties_by_tier_prefix']['LARGE'].items():
        lines.append('| M'+n+' | '+' | '.join(str(families[f]) for f in FAMILIES)+' |')
    lines += ['', '## 候选与状态计数', '',
        f"24 条原生轨迹全部完成，{o['states']} 个状态 M64 exact replay 全部通过；{result['costs']['candidates']:,} 个 valid unique M256 候选全部完成标签。独立候选标签："+dumps(result['candidate_status_counts'])+f"；NUMERIC_FAILURE={result['numeric_failures']}、certifier mismatch={result['certifier_mismatches']}。其他已声明状态计数为 0。", '',
        '| prefix | 有效 unique 总数 | 每状态平均 | 每状态中位 |', '|---|---:|---:|---:|']
    for n in (64,128,256):
        count=sum(r['unique_counts'][str(n)] for r in result['states'])
        v=o['unique_'+str(n)]
        lines.append(f"| M{n} | {count} | {v['mean']:.3f} | {v['median']:.3f} |")
    lines += ['', '尝试级状态计数（每状态固定 64/128/256；重复/identity/拒绝也占预算）：', '',
        '| status | M64 | M128 | M256 |', '|---|---:|---:|---:|']
    for status in result['attempt_status_counts']['64']:
        lines.append('| '+status+' | '+' | '.join(str(result['attempt_status_counts'][str(n)][status]) for n in (64,128,256))+' |')
    lines += ['', '## 成本与复现', '', '```json',json.dumps(result['costs'],indent=2),'```','',
        '无损存储优化：'+dumps(result.get('storage_compaction',{})), '',
        '压缩块按状态保存，标量标签继续逐候选索引。可用审计脚本的 read_candidate_payload、read_candidate_diagnostic、read_attempt_detail 读取完整记录；传入对应实例 parents 即可恢复共享几何。完整身份通过 expanded_identity 展开，未替换为摘要或截断标识。', '',
        '轨迹的原生 60 秒预算与离线 replay/oracle 计时分离。离线 oracle 使用 3 个独立进程，主进程按确定顺序逐候选写入；candidate wall/scheduler time 为调用耗时之和，offline_elapsed_wall_seconds 为并行阶段实际经过时间。候选标签按协议、上下文、完整候选身份、scope 和源码身份联合主键逐条提交事务，可断点续跑；只读取已完成且身份匹配的标签。状态 JSON 共享几何并使用无损压缩；SQLite 候选与诊断也使用无损压缩。', '',
        'FEASIBLE_CERTIFIED 才有有限 Cmax；其他状态 Cmax/delta 为 NULL。无可行候选时 C* 为 NULL、I 为零并保留 no_feasible 标记。无改善机会时 recall 为 NA。排名 recall 仅在 I64>epsilon 状态上聚合；generation recall 仅在 I256>epsilon 状态上聚合。', '',
        '辅助的 C4/C2 条件 recall：'+dumps(result['c4_conditional_c2_recall_auxiliary'])+'；不替换以 I64 为分母的主指标。POST_C4_DIRECTION_GAIN：'+dumps(result['post_c4_direction_gain'])+'，仅单独描述方向细化。', '',
        '候选标签文件中的 production_reference_evaluated 与 audit_oracle_reference_evaluated 分列。方向细化收益在每状态 POST_C4_DIRECTION_GAIN 列中，不进入基础 oracle 排名。', '',
        '历史 Phase3-Z/Phase3-ZR 文件不改写。V2 validation 已消耗，ID_TEST 继续封存。本轮仅运行 RANKER_DEV；路径含 VALIDATION 的文件按既有 manifest 角色判定，文件夹名字不决定角色。', '',
        'INITIAL 回退状态数：'+str(result['initial_stage_fallbacks'])+' / 72。', '',
        '回归验证：'+dumps(result.get('regression',{})), '源身份：'+dumps(result['source']), '',
        '协议与科学身份：'+dumps({k:result[k] for k in ('protocol_hash','ranker_split_hash','scope_hash','scientific_config_hash','reference_scheduler_policy_id')})]
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
