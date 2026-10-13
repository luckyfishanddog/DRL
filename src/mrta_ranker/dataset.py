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
from mrta_reference.model import CanonicalSolution,ScientificConfig,ScheduleStatus,SplitKind
from mrta_reference.geometry import build_legal_pattern_catalog
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
DATABASE_V05=ROOT/'data/development/graph_ranker_dataset_v05.sqlite'
RESULT=ROOT/'data/development/phase4_1b_mlp_results.json'
STAGES=(('EARLY',5.0),('MID',30.0),('LATE',60.0))
SEEDS=(20261121,20261122)
TIERS=('SMALL','MEDIUM','LARGE')
# Phase4-1A/B/C labels and checkpoints belong to the historical 0.2 domain.
CONFIG=ScientificConfig(delta_x=0.20,delta_y=0.20)
CONFIG_V05=ScientificConfig()
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


def representatives_v05(split=None,manifest=None):
    """Workbook-isolated 35/55/85 selection using static weld counts only."""
    split=validate_split(read(SPLIT) if split is None else split)
    manifest=read(DATASET) if manifest is None else manifest
    targets=(35,55,85);selected=[]
    for role,counts in (('MLP_TRAIN',(6,6,5)),('MLP_DEV',(2,2,2))):
        paths=sorted(split[role])
        available={p:[e for e in manifest['instances'] if e['relative_path']==p and e['validation_status']=='VALID' and e['duplicate_of'] is None and 10<=e['actual_weld_count']<=90] for p in paths}
        if any(not rows for rows in available.values()): raise ValueError('Workbook has no static eligible instance')
        dp={(0,0,0):(0,())}
        for path in paths:
            choices=[min(available[path],key=lambda e:(abs(e['actual_weld_count']-target),e['sheet_name'],e['instance_id'])) for target in targets]
            nxt={}
            for used,(cost,assignment) in dp.items():
                for t,entry in enumerate(choices):
                    if used[t]>=counts[t]: continue
                    capacity=tuple(n+int(i==t) for i,n in enumerate(used))
                    proposal=(cost+abs(entry['actual_weld_count']-targets[t]),assignment+(t,))
                    if capacity not in nxt or proposal<nxt[capacity]: nxt[capacity]=proposal
            dp=nxt
        assignment=dp[counts][1]
        for path,t in zip(paths,assignment):
            e=min(available[path],key=lambda row:(abs(row['actual_weld_count']-targets[t]),row['sheet_name'],row['instance_id']))
            selected.append({**{k:e[k] for k in ('instance_id','relative_path','sheet_name','raw_file_sha256','instance_geometry_hash')},
                'split':role,'target_tier':TIERS[t],'tier':TIERS[t],'N':e['actual_weld_count'],'target_N':targets[t]})
    return selected


def static_geometry_summary(parents,config=CONFIG_V05):
    """Static-only selection descriptors; never evaluates a solution or schedule."""
    catalog=build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)
    xs=[v for p in parents for v in (p.start[0],p.end[0])]
    ys=[v for p in parents for v in (p.start[1],p.end[1])]
    return {'total_weld_length':sum(p.length for p in parents),'x_span':max(xs)-min(xs),'y_span':max(ys)-min(ys),
        'x_split_opportunities':sum(any(q.kind is SplitKind.X_SPLIT for q in catalog[p.parent_id]) for p in parents),
        'y_split_opportunities':sum(any(q.kind is SplitKind.Y_SPLIT for q in catalog[p.parent_id]) for p in parents)}


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
    def __init__(self):
        self.snapshots={};self.first_boundary_elapsed=None;self.initial=None;self.best60=None
        self.crrr_examples=[]
    def __call__(self,event,value):
        if event=='crrr_attempt':
            _,repaired,status=value
            if repaired.candidate is not None and len(self.crrr_examples)<16:
                self.crrr_examples.append({
                    'status':status,
                    'provenance':dict(repaired.candidate.provenance),
                    'routes':[list(route.block_ids) for route in repaired.candidate.solution.routes],
                })
            return
        if event!='boundary': return
        if self.first_boundary_elapsed is None: self.first_boundary_elapsed=value['elapsed']
        if 'metrics' in value:
            if self.initial is None: self.initial=deepcopy(value)
            if value['elapsed']<=60.0 and (self.best60 is None or value['metrics'].cmax<self.best60['metrics'].cmax): self.best60=deepcopy(value)
        eligible=[stage for stage,target in STAGES if value['elapsed']<=target]
        if eligible:
            snapshot=dict(value);snapshot['adaptive']=deepcopy(value['adaptive'])
            for stage in eligible: self.snapshots[stage]=snapshot


def checkpoint_cmax(result): return result.anytime[60.0]['cmax']


def strict_cmax_improvement_count(events,deadline=60.0):
    """Count only strict Cmax decreases, excluding equal-Cmax metric updates."""
    eligible=[(elapsed,cmax) for elapsed,cmax in events if elapsed<=deadline]
    if not eligible:return 0
    incumbent=eligible[0][1];count=0
    for _,cmax in eligible[1:]:
        if cmax<incumbent-1e-9*max(1.0,abs(incumbent)):
            count+=1;incumbent=cmax
    return count


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


def run_trajectory(entry,seed,*,parents=None,source=None,ranker=None,config=None,search=None):
    config=CONFIG if config is None else config
    parents=load_parents(entry) if parents is None else parents
    cap=Capture()
    with generation_timers() as timings:
        result=pipeline.run_sa_oi_alns_v2(parents,config,search or search_config(),seed=seed,source_provenance=source or provenance(),observer=cap,**({'ranker':ranker} if ranker is not None else {}))
    st=result.stats
    from mrta_reference.solution import block_map
    best=cap.best60;blocks=None if best is None else block_map(best['solution'],config)
    checkpoint_certification=(
        None if best is None
        else certify_schedule(best['solution'],best['schedule'],config,scope=FORMAL_SCOPE_V2)
    )
    loads=None if best is None else [sum(config.process_time(blocks[b].length) for b in route.block_ids) for route in best['solution'].routes]
    initial_blocks=None if cap.initial is None else block_map(cap.initial['solution'],config)
    initial_loads=(None if cap.initial is None else [
        sum(config.process_time(initial_blocks[b].length) for b in route.block_ids)
        for route in cap.initial['solution'].routes
    ])
    empty_by_robot=waiting_by_robot=None
    if best is not None:
        from mrta_reference.model import OperationKind
        empty_by_robot=[];waiting_by_robot=[]
        for robot in range(4):
            welds=sorted((op for op in best['schedule'].operations if op.robot_id==robot and op.kind is OperationKind.WELD),key=lambda op:op.sequence_index)
            empty_by_robot.append(sum(math.dist(left.end,right.start)/config.empty_speed for left,right in zip(welds,welds[1:])))
            waiting_by_robot.append(sum(op.duration for op in best['schedule'].operations if op.robot_id==robot and op.kind is OperationKind.WAIT))
    split_counts=None if best is None else dict(Counter(pattern.kind.value for pattern in best['solution'].patterns))
    strict_improvements=strict_cmax_improvement_count(st.best_events)
    record={'initial_cmax':None if cap.initial is None else cap.initial['metrics'].cmax,
        'initial_process_loads':initial_loads,
        'initial_empty_travel':None if cap.initial is None else cap.initial['metrics'].total_empty_travel,
        'initial_waiting':None if cap.initial is None else cap.initial['metrics'].total_waiting,
        'initialization_seconds':st.init_time,
        'initialization_strategy':result.initialization.winning_strategy,
        'Cmax_at_60':checkpoint_cmax(result),'final_cmax':None if result.best_metrics is None else result.best_metrics.cmax,
        'certified':bool(result.final_certification and result.final_certification.certified),'iterations':st.iterations,
        'accepted_moves':sum(st.accepted_by_family.values()),'global_best_updates':len(st.best_improvement_cmax),
        'reference_calls':st.nref+st.init_reference_calls,'reference_seconds':st.reference_scheduler_time,
        'direction_calls':sum(st.per_iteration_kdp)+st.repair_full_direction_dp_calls,'direction_seconds':st.direction_dp_time,
        'reference_status':{'FEASIBLE':st.n_feasible,'DEADLOCK':st.n_deadlock,'INFEASIBLE':st.n_infeasible,'NUMERIC_FAILURE':st.n_numeric_failure},
        'direction_feasible':sum(st.decision_family_funnel[f]['direction_feasible'] for f in pipeline.DECISION_FAMILIES),
        'c2_selected':st.kdp_count,'best60_process_loads':loads,
        'c2_selection_seconds':st.c2_selection_time,
        'c4_selection_seconds':st.c4_selection_time,
        'best60_empty_travel':None if best is None else best['metrics'].total_empty_travel,
        'best60_waiting':None if best is None else best['metrics'].total_waiting,
        'best60_empty_travel_by_robot':empty_by_robot,'best60_waiting_by_robot':waiting_by_robot,
        'best60_split_counts':split_counts,
        'numeric_failure':st.n_numeric_failure+st.init_status_counts.get('NUMERIC_FAILURE',0),
        'actual_runtime':result.runtime,'overshoot':st.overshoot,**timings,
        'ranker_inference_seconds':0.0 if ranker is None else ranker.inference_seconds,
        'graph_build_seconds':0.0 if ranker is None else getattr(ranker,'graph_build_seconds',0.0),
        'static_graph_forward_seconds':0.0 if ranker is None else getattr(ranker,'static_forward_seconds',0.0),
        'candidate_scoring_seconds':0.0 if ranker is None else getattr(ranker,'candidate_scoring_seconds',0.0),
        'ranking_seconds':0.0 if ranker is None else getattr(ranker,'ranking_seconds',0.0),
        'c2_ranker_seconds':0.0 if ranker is None else getattr(ranker,'c2_seconds',0.0),
        'c4_ranker_seconds':0.0 if ranker is None else getattr(ranker,'c4_seconds',0.0),
        'best_events':st.best_events,'raw_attempts':st.raw_attempts,'valid_unique':sum(st.valid_by_family.values()),
        'attempted_by_family':dict(st.attempted_by_family),
        'max_per_iteration_attempts':max(st.per_iteration_attempts,default=0),
        'strict_best_improvements_at60':strict_improvements,
        'crrr_funnel':{
            'attempted':st.crrr_attempted,'constructed':st.crrr_constructed,
            'valid_unique':st.crrr_valid_unique,'c2_selected':st.crrr_c2_selected,
            'direction_feasible':st.crrr_direction_feasible,'c4_selected':st.crrr_c4_selected,
            'reference_evaluated':st.crrr_reference_evaluated,
            'reference_status':dict(st.crrr_reference_status),
            'accepted':st.crrr_accepted,
            'strict_cmax_improvements':st.crrr_strict_cmax_improvements,
            'competitive':st.crrr_access_competitive,
            'rejected_by_access_gate':st.crrr_access_rejected,
            'injected':st.crrr_access_injected,
            'rejection_reasons':dict(st.crrr_rejection_reasons),
        },
        'source_selection_funnel':{
            'atomic_c2_selected':st.atomic_c2_selected,
            'atomic_c4_selected':st.atomic_c4_selected,
            'original_lns_c2_selected':st.original_lns_c2_selected,
            'original_lns_c4_selected':st.original_lns_c4_selected,
        },
        'crrr_generation_seconds':st.crrr_generation_time,
        'search_repair_seconds':st.repair_time,
        'crrr_examples':cap.crrr_examples,
        'initial_reference_calls':st.init_reference_calls,'search_reference_calls':st.nref,
        'initial_reference_status':dict(st.init_status_counts),
        'search_reference_status':dict(Counter(st.reference_status_sequence)),
        'status':result.status.value,'termination_reason':result.termination_reason,
        'certified_at_60':bool(checkpoint_certification and checkpoint_certification.certified),
        'checkpoint_certification_errors':([] if checkpoint_certification is None else list(checkpoint_certification.errors)),
        'first_boundary_elapsed':cap.first_boundary_elapsed,
        'time_to_first_certified':None if cap.initial is None else cap.initial['elapsed'],
        'missing_state_stages':[stage for stage,_ in STAGES if stage not in cap.snapshots]}
    record['valid_60s_outcome']=not record['numeric_failure'] and record['certified_at_60'] and record['Cmax_at_60'] is not None
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
        CREATE TABLE IF NOT EXISTS experiments(experiment TEXT NOT NULL,item_id TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(experiment,item_id));
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


def generate_state_pool(db,state,config=None):
    config=CONFIG if config is None else config
    context=state_context(db,state);parents=context['solution'].parents
    stats=SearchStats(FORMAL_SCOPE_V2.scope_id,state['seed']);stats.iterations=context['iteration']
    _,pool,_=pipeline.generate_candidate_pool(context['solution'],context['directions'],config,search_config(),stats,
        seed=state['seed']+context['iteration']*65537,current_schedule=context['schedule'],adaptive_state=deepcopy(context['adaptive']),scope=FORMAL_SCOPE_V2,enable_x_split=True)
    pool=sorted(pool,key=lambda c:c.cheap_score)
    if (stats.raw_attempts,stats.attempted_by_family['ATOMIC'],stats.attempted_by_family['LNS_REPAIRED'])!=(192,144,48): raise ValueError('Expected 144 atomic + 48 LNS attempts')
    with db:
        for rank,candidate in enumerate(pool):
            features=extract_c2_features(context['solution'],context['directions'],state['Cs'],candidate,config)
            old=metadata(db,'C2_feature_names')
            if old is not None and old!=list(features): raise ValueError('C2 schema drift')
            if old is None: set_meta(db,'C2_feature_names',list(features))
            db.execute('INSERT INTO candidates(state_id,candidate_id,family,payload,c2) VALUES (?,?,?,?,?)',(state['state_id'],rank,pipeline.decision_family(candidate),pack(candidate,parents),feature_blob(features)))
        db.execute('UPDATE states SET pool_complete=1 WHERE state_id=?',(state['state_id'],))
    return len(pool)


def label_candidate(candidate,context,cheap_rank,config=None):
    """One real direction DP and at most one B32 reference evaluation."""
    config=CONFIG if config is None else config
    start=time.perf_counter();direction=optimize_directions_with_initial_feasibility(candidate.solution,config)
    dp_seconds=time.perf_counter()-start;cmax=None;ref_seconds=0.0;c4=None
    if direction.total_empty_travel is None:
        status='DIRECTION_INFEASIBLE'
    else:
        c4_features=extract_c4_features(context['solution'],context['directions'],context['metrics'].cmax,candidate,config,direction=direction,cheap_rank=cheap_rank)
        c4=feature_blob(c4_features)
        start=time.perf_counter()
        schedule=FormalReferenceEvaluator(FORMAL_SCOPE_V2)(candidate.solution,config,orientations={r:direction.directions[r] for r in range(4)})
        ref_seconds=time.perf_counter()-start;status=schedule.status.value
        if status=='FEASIBLE':
            certification=certify_schedule(candidate.solution,schedule,config,scope=FORMAL_SCOPE_V2)
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


def material_positive(row):
    """Existing reference labels only; numeric failures are excluded by the caller."""
    return int(row['status']=='FEASIBLE_CERTIFIED' and
               (row['Cs']-row['reference_cmax'])/row['Cs']>=0.005)


def c4_hard_subset(pool,baseline_top8,new_top8,legacy_top8,heuristic_key):
    """In-memory union for one TRAIN state; never evaluates new candidates."""
    eligible=[r for r in pool if r['C4_features'] is not None and r['status']!='DIRECTION_INFEASIBLE']
    ids={r['candidate_id'] for r in (*baseline_top8,*new_top8,*legacy_top8)}
    ids.update(r['candidate_id'] for r in eligible if material_positive(r))
    extra=[r for r in sorted(eligible,key=heuristic_key) if r['candidate_id'] not in ids][:8]
    ids.update(r['candidate_id'] for r in extra)
    return [r for r in eligible if r['candidate_id'] in ids]


def hard_ranking_pairs(rows,stage,heuristic_key):
    """Up to four negatives per positive, prioritizing heuristic and V1 tails.

    Alternating both rankings preserves contributions from both failure sources.
    Return row indices and state-equal weights; no cross-state comparisons.
    """
    from collections import defaultdict
    states=defaultdict(list)
    for i,r in enumerate(rows): states[r['state_id']].append(i)
    by_state=[]
    for indices in states.values():
        positives=[i for i in indices if material_positive(rows[i])]
        negatives=[i for i in indices if not material_positive(rows[i])]
        a=sorted(negatives,key=lambda i:heuristic_key(rows[i]))
        b=sorted(negatives,key=lambda i:(-rows[i][stage+'_v1_score'],heuristic_key(rows[i])))
        chosen=[]
        for j in range(len(negatives)):
            for order in (a,b):
                if order[j] not in chosen: chosen.append(order[j])
                if len(chosen)==4: break
            if len(chosen)==4: break
        pairs=[(p,n) for p in positives for n in chosen]
        if pairs: by_state.append(pairs)
    return [(p,n,1/(len(by_state)*len(pairs))) for pairs in by_state for p,n in pairs]
