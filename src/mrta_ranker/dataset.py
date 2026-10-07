"""Workbook-isolated candidate data, accepted-state observation and typed storage."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields,is_dataclass,replace
from enum import Enum
import ctypes,importlib,json,math,sqlite3,statistics,struct,time,zlib
from ctypes import wintypes
from pathlib import Path
from mrta_reference.model import CanonicalSolution,ScientificConfig,ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.provenance import SourceProvenance,REPOSITORY_ID,compute_source_tree_hash
from mrta_reference.scheduler import FormalReferenceEvaluator
from mrta_reference.certifier import certify_schedule
from mrta_data.ppo_instances import load_ppo_platform_instance,instance_geometry_hash,to_parent_welds
from mrta_data.phase3_split import assert_v2_solver_access_allowed
from mrta_search import pipeline
from mrta_search.stats import SearchStats
from mrta_search.direction import optimize_directions_with_initial_feasibility
from .features import extract_c2_features,extract_c4_features,make_training_targets
ROOT=Path(__file__).resolve().parents[2]
PPO=ROOT.parent/'ppo'
SPLIT=ROOT/'data/development/mlp_workbook_split_v1.json'
DATASET=ROOT/'data/manifests/PPO_DATASET_MANIFEST_V1.json'
ROLES=ROOT/'data/manifests/PPO_V2_DATA_ROLES_V1.json'
DATABASE=ROOT/'data/development/mlp_candidate_dataset_v1.sqlite'
RESULT=ROOT/'data/development/phase4_1b_mlp_results.json'
STAGES=(('EARLY',5.0),('MID',30.0),('LATE',60.0))
SEEDS=(20261121,20261122)
TIERS=('SMALL','MEDIUM','LARGE')
CONFIG=ScientificConfig()
def dumps(value): return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def provenance(): return SourceProvenance(REPOSITORY_ID,'LOCAL_DRL_USER_AUTHORIZED',compute_source_tree_hash(ROOT),True,False)
def search_config():
    return pipeline.SearchConfig(m=192,m_lns=48,construction_budget=5,kinit_ref=5,max_iterations=100000,time_limit=60.0,checkpoints=(5.0,30.0,60.0),enable_two_opt_star=False)

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

def validate_split(split):
    groups=[set(split[k]) for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED')]
    if tuple(len(split[k]) for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED'))!=(17,6,8) or tuple(map(len,groups))!=(17,6,8) or any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Expected disjoint 17/6/8 workbook sets')
    return split


def representatives(split=None,manifest=None):
    """One static representative per workbook; tier assignment minimizes N distance."""
    split=validate_split(read(SPLIT) if split is None else split)
    manifest=read(DATASET) if manifest is None else manifest
    selected=[]
    for role,counts in (('MLP_TRAIN',(6,6,5)),('MLP_DEV',(2,2,2))):
        paths=sorted(split[role]);available={p:[e for e in manifest['instances'] if e['relative_path']==p and e['validation_status']=='VALID' and e['duplicate_of'] is None and 10<=e['actual_weld_count']<=90] for p in paths}
        if any(not rows for rows in available.values()): raise ValueError('Workbook has no static eligible instance')
        # Deterministic minimum-cost assignment of target tiers to workbooks.
        # Enumerating capacity states (<=7*7*6) avoids arbitrary path-order biases.
        dp={(0,0,0):(0,())}
        for path in paths:
            options=[]
            for tier,target in zip(TIERS,(25,55,85)):
                entry=min(available[path],key=lambda e:(abs(e['actual_weld_count']-target),e['sheet_name'],e['instance_id']))
                options.append(entry)
            nxt={}
            for used,(cost,assignment) in dp.items():
                for t,entry in enumerate(options):
                    if used[t]>=counts[t]: continue
                    capacity=tuple(n+int(i==t) for i,n in enumerate(used))
                    proposal=(cost+abs(entry['actual_weld_count']-(25,55,85)[t]),assignment+(t,))
                    if capacity not in nxt or proposal<nxt[capacity]: nxt[capacity]=proposal
            dp=nxt
        assignment=dp[counts][1]
        for path,t in zip(paths,assignment):
            target=(25,55,85)[t]
            e=min(available[path],key=lambda e:(abs(e['actual_weld_count']-target),e['sheet_name'],e['instance_id']))
            n=e['actual_weld_count'];tier=min(range(3),key=lambda j:abs(n-(25,55,85)[j]))
            selected.append({**{k:e[k] for k in ('instance_id','relative_path','sheet_name','raw_file_sha256','instance_geometry_hash')},'split':role,'target_tier':TIERS[t],'tier':TIERS[tier],'N':n})
    return selected


def load_parents(entry,*,split=None,loader=None):
    split=validate_split(read(SPLIT) if split is None else split)
    if entry.get('split') not in ('MLP_TRAIN','MLP_DEV') or entry['relative_path'] not in split[entry['split']]:
        raise PermissionError('Workbook is outside the explicit MLP split whitelist')
    assert_v2_solver_access_allowed(read(ROLES),[entry['relative_path']],allowed_roles=('V2_TRAIN_POOL',))
    instance=(loader or load_ppo_platform_instance)(PPO/entry['relative_path'],entry['sheet_name'],ppo_root=PPO,instance_id=entry['instance_id'])
    if instance.raw_file_sha256!=entry['raw_file_sha256'] or instance_geometry_hash(instance)!=entry['instance_geometry_hash']:
        raise ValueError('Workbook/geometry differs from existing dataset identity')
    return to_parent_welds(instance)


class Capture:
    """Last certified accepted-state boundary at or before each target; no future backfill."""
    def __init__(self): self.snapshots={};self.first_boundary_elapsed=None
    def __call__(self,event,value):
        if event!='boundary': return
        if self.first_boundary_elapsed is None: self.first_boundary_elapsed=value['elapsed']
        eligible=[stage for stage,target in STAGES if value['elapsed']<=target]
        if eligible:
            snapshot=dict(value);snapshot['adaptive']=deepcopy(value['adaptive'])
            for stage in eligible: self.snapshots[stage]=snapshot


def checkpoint_cmax(result): return result.anytime[60.0]['cmax']


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
    for name in originals: setattr(pipeline,name,timed(name,'candidate_generation_seconds' if name=='generate_candidate_pool' else 'lns_seconds'))
    try: yield totals
    finally:
        for name,fn in originals.items(): setattr(pipeline,name,fn)


def run_trajectory(entry,seed,*,parents=None,source=None,ranker=None):
    parents=load_parents(entry) if parents is None else parents
    cap=Capture()
    with generation_timers() as timings:
        result=pipeline.run_sa_oi_alns_v2(parents,CONFIG,search_config(),seed=seed,source_provenance=source or provenance(),observer=cap,**({'ranker':ranker} if ranker is not None else {}))
    st=result.stats
    record={'Cmax_at_60':checkpoint_cmax(result),'final_cmax':None if result.best_metrics is None else result.best_metrics.cmax,
        'certified':bool(result.final_certification and result.final_certification.certified),'iterations':st.iterations,
        'reference_calls':st.nref+st.init_reference_calls,'reference_seconds':st.reference_scheduler_time,
        'numeric_failure':st.n_numeric_failure+st.init_status_counts.get('NUMERIC_FAILURE',0),
        'actual_runtime':result.runtime,'overshoot':st.overshoot,**timings,
        'ranker_inference_seconds':0.0 if ranker is None else ranker.inference_seconds,
        'best_events':st.best_events,'raw_attempts':st.raw_attempts,'valid_unique':sum(st.valid_by_family.values()),
        'status':result.status.value,'termination_reason':result.termination_reason,
        'first_boundary_elapsed':cap.first_boundary_elapsed,
        'missing_state_stages':[stage for stage,_ in STAGES if stage not in cap.snapshots]}
    record['valid_60s_outcome']=not record['numeric_failure'] and record['certified'] and record['Cmax_at_60'] is not None
    return record,cap.snapshots


def connect(path=DATABASE,*,readonly=False):
    if readonly:
        db=sqlite3.connect('file:'+Path(path).as_posix()+'?mode=ro',uri=True)
    else:
        db=sqlite3.connect(path,timeout=60);db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS instances(instance_id TEXT PRIMARY KEY,split TEXT NOT NULL,workbook TEXT UNIQUE NOT NULL,entry TEXT NOT NULL,parents BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS trajectories(instance_id TEXT NOT NULL REFERENCES instances,seed INTEGER NOT NULL,result TEXT NOT NULL,PRIMARY KEY(instance_id,seed));
        CREATE TABLE IF NOT EXISTS states(state_id TEXT PRIMARY KEY,instance_id TEXT NOT NULL REFERENCES instances,seed INTEGER NOT NULL,stage TEXT NOT NULL,tier TEXT NOT NULL,iteration INTEGER NOT NULL,elapsed REAL NOT NULL,Cs REAL NOT NULL,context BLOB NOT NULL,pool_complete INTEGER NOT NULL DEFAULT 0,UNIQUE(instance_id,seed,stage));
        CREATE TABLE IF NOT EXISTS candidates(state_id TEXT NOT NULL REFERENCES states,candidate_id INTEGER NOT NULL,family TEXT NOT NULL,payload BLOB NOT NULL,c2 BLOB NOT NULL,c4 BLOB,direction BLOB,status TEXT,y_feasible INTEGER,y_improvement REAL,reference_cmax REAL,direction_seconds REAL,reference_seconds REAL,PRIMARY KEY(state_id,candidate_id));
        CREATE VIEW IF NOT EXISTS candidate_records AS SELECT c.*,i.split,i.workbook,s.instance_id,s.seed,s.stage,s.tier,s.iteration,s.Cs FROM candidates c JOIN states s USING(state_id) JOIN instances i USING(instance_id);
        ''')
    db.row_factory=sqlite3.Row
    return db


def feature_blob(features): return struct.pack('<'+str(len(features))+'f',*features.values())
def set_meta(db,key,value): db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,dumps(value)))
def metadata(db,key):
    row=db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    return None if row is None else json.loads(row[0])


def store_trajectory(db,entry,parents,seed,record,snapshots):
    with db:
        db.execute('INSERT OR IGNORE INTO instances VALUES (?,?,?,?,?)',(entry['instance_id'],entry['split'],entry['relative_path'],dumps(entry),pack(parents)))
        db.execute('INSERT INTO trajectories VALUES (?,?,?)',(entry['instance_id'],seed,dumps(record)))
        for stage,ctx in snapshots.items():
            state_id=f"{entry['instance_id']}:{seed}:{stage}"
            db.execute('INSERT INTO states(state_id,instance_id,seed,stage,tier,iteration,elapsed,Cs,context) VALUES (?,?,?,?,?,?,?,?,?)',
                (state_id,entry['instance_id'],seed,stage,entry['tier'],ctx['iteration'],ctx['elapsed'],ctx['metrics'].cmax,pack(ctx,parents)))


def state_context(db,state):
    parents=unpack(db.execute('SELECT parents FROM instances WHERE instance_id=?',(state['instance_id'],)).fetchone()[0])
    return unpack(state['context'],parents)


def generate_state_pool(db,state):
    context=state_context(db,state);parents=context['solution'].parents
    stats=SearchStats(FORMAL_SCOPE_V2.scope_id,state['seed']);stats.iterations=context['iteration']
    _,pool,_=pipeline.generate_candidate_pool(context['solution'],context['directions'],CONFIG,search_config(),stats,
        seed=state['seed']+context['iteration']*65537,current_schedule=context['schedule'],adaptive_state=deepcopy(context['adaptive']),scope=FORMAL_SCOPE_V2,enable_x_split=True)
    pool=sorted(pool,key=lambda c:c.cheap_score)
    if (stats.raw_attempts,stats.attempted_by_family['ATOMIC'],stats.attempted_by_family['LNS_REPAIRED'])!=(192,144,48): raise ValueError('Expected 144 atomic + 48 LNS attempts')
    with db:
        for rank,candidate in enumerate(pool):
            features=extract_c2_features(context['solution'],context['directions'],state['Cs'],candidate,CONFIG)
            old=metadata(db,'C2_feature_names')
            if old is not None and old!=list(features): raise ValueError('C2 schema drift')
            if old is None: set_meta(db,'C2_feature_names',list(features))
            db.execute('INSERT INTO candidates(state_id,candidate_id,family,payload,c2) VALUES (?,?,?,?,?)',(state['state_id'],rank,pipeline.decision_family(candidate),pack(candidate,parents),feature_blob(features)))
        db.execute('UPDATE states SET pool_complete=1 WHERE state_id=?',(state['state_id'],))
    return len(pool)


def label_candidate(candidate,context,cheap_rank):
    """One real direction DP and at most one B32 reference evaluation."""
    start=time.perf_counter();direction=optimize_directions_with_initial_feasibility(candidate.solution,CONFIG)
    dp_seconds=time.perf_counter()-start;cmax=None;ref_seconds=0.0;c4=None
    if direction.total_empty_travel is None:
        status='DIRECTION_INFEASIBLE'
    else:
        c4_features=extract_c4_features(context['solution'],context['directions'],context['metrics'].cmax,candidate,CONFIG,direction=direction,cheap_rank=cheap_rank)
        c4=feature_blob(c4_features)
        start=time.perf_counter()
        schedule=FormalReferenceEvaluator(FORMAL_SCOPE_V2)(candidate.solution,CONFIG,orientations={r:direction.directions[r] for r in range(4)})
        ref_seconds=time.perf_counter()-start;status=schedule.status.value
        if status=='FEASIBLE':
            certification=certify_schedule(candidate.solution,schedule,CONFIG,scope=FORMAL_SCOPE_V2)
            if certification.certified and schedule.cmax is not None and math.isfinite(schedule.cmax): status='FEASIBLE_CERTIFIED';cmax=schedule.cmax
            else: status='NUMERIC_FAILURE'
    targets=make_training_targets(status,context['metrics'].cmax,cmax)
    return {'status':status,'reference_cmax':cmax,'y_feasible':None if targets is None else targets['y_feasible'],
        'y_improvement':None if targets is None else targets['y_improvement'],'c4':c4,'direction':pack(direction),
        'direction_seconds':dp_seconds,'reference_seconds':ref_seconds,
        'c4_names':None if c4 is None else list(c4_features)}


def dataset_summary(db):
    out={}
    for split in ('MLP_TRAIN','MLP_DEV'):
        states=db.execute('SELECT count(*) FROM states s JOIN instances i USING(instance_id) WHERE i.split=?',(split,)).fetchone()[0]
        counts=dict(db.execute('SELECT c.status,count(*) FROM candidates c JOIN states s USING(state_id) JOIN instances i USING(instance_id) WHERE i.split=? GROUP BY c.status',(split,)))
        out[split]={'states':states,'candidates':sum(counts.values()),'status_counts':counts}
    return out


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
