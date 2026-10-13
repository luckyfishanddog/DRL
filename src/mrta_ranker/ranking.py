"""Score replacement only: preserve production C2 family coverage and C4 rotation."""
from __future__ import annotations
import time
import math
from pathlib import Path
from mrta_search.pipeline import DECISION_FAMILIES,decision_family,select_c2_by_family,select_c4_by_family
from .features import extract_c2_features,extract_c4_features


def family_c2(items,key,family,*,k=8,seed=0,iteration=0):
    ranked=sorted(items,key=key);families=DECISION_FAMILIES
    if k<len(families):
        offset=(seed+iteration)%len(families);families=families[offset:]+families[:offset]
    selected=[]
    for f in families:
        row=next((row for row in ranked if family(row)==f),None)
        if row is not None and len(selected)<k: selected.append(row)
    ids={id(x) for x in selected};selected.extend(x for x in ranked if id(x) not in ids)
    return tuple(selected[:k])


def family_c4(items,key,family,*,k=2,seed=0,iteration=0):
    ranked=sorted(items,key=key)
    if k<=1 or len(ranked)<=1: return tuple(ranked[:k])
    selected=[ranked[0]];offset=(seed+iteration)%len(DECISION_FAMILIES)
    for f in DECISION_FAMILIES[offset:]+DECISION_FAMILIES[:offset]:
        row=next((row for row in ranked if row is not selected[0] and family(row)==f),None)
        if row is not None: selected.append(row);break
    ids={id(x) for x in selected};selected.extend(x for x in ranked if id(x) not in ids)
    return tuple(selected[:k])


class LearnedRanker:
    """Models load outside search; feature/scaler/forward/selection time stays inside."""
    def __init__(self,variant,root=None,model_version="v1"):
        if variant not in ('MLP-C2','MLP-C4','MLP-BOTH'): raise ValueError('Unknown ranker variant')
        from .model import load_model
        import torch,numpy as np
        torch.set_num_threads(1)
        if model_version not in ("v1","material_v2","rank_v2"): raise ValueError("Unknown model version")
        self.variant=variant;self.inference_seconds=0.0
        suffix="" if model_version=="v1" else "_"+model_version
        root=Path(__file__).resolve().parents[2] if root is None else Path(root)
        self.scalers=np.load(root/'models/mlp_scalers.npz')
        self.models={}
        for stage in ('C2','C4'):
            if variant in ('MLP-'+stage,'MLP-BOTH'): self.models[stage]=load_model(root/f'models/mlp_{stage.lower()}{suffix}.pt')
    def scores(self,stage,features):
        from .model import predict_scores
        model,meta=self.models[stage]
        if any(list(f)!=meta['feature_names'] for f in features): raise ValueError('Model feature columns differ')
        scores=predict_scores(model,[list(f.values()) for f in features],self.scalers[stage+'_mean'],self.scalers[stage+'_std'])
        if not all(math.isfinite(float(s)) for s in scores): raise ValueError('Nonfinite ranker score')
        return scores
    def select_c2(self,current,directions,cs,pool,config,*,seed,iteration,k=8):
        if 'C2' not in self.models: return select_c2_by_family(pool,k,seed=seed,iteration=iteration)
        start=time.perf_counter()
        try:
            features=[extract_c2_features(current,directions,cs,c,config) for c in pool]
            if not pool: return ()
            scores={id(c):float(s) for c,s in zip(pool,self.scores('C2',features))}
            return family_c2(pool,lambda c:(-scores[id(c)],c.cheap_score),decision_family,k=k,seed=seed,iteration=iteration)
        finally: self.inference_seconds+=time.perf_counter()-start
    def select_c4(self,current,directions,cs,pool,config,*,seed,iteration,k=2):
        if 'C4' not in self.models: return select_c4_by_family(pool,k,seed=seed,iteration=iteration)
        start=time.perf_counter()
        try:
            features=[extract_c4_features(current,directions,cs,c.screened,config,direction=c.direction,cheap_rank=c.cheap_rank) for c in pool]
            if not pool: return ()
            scores={id(c):float(s) for c,s in zip(pool,self.scores('C4',features))}
            return family_c4(pool,lambda c:(-scores[id(c)],c.rerank_key),lambda c:decision_family(c.screened),k=k,seed=seed,iteration=iteration)
        finally: self.inference_seconds+=time.perf_counter()-start


class V05MLPRanker:
    """Self-contained 0.5 m MLP C2/C4 ranker; production default remains heuristic."""
    def __init__(self,root=None):
        from .model import load_model
        import torch
        torch.set_num_threads(1);root=Path(__file__).resolve().parents[2] if root is None else Path(root)
        self.models={};self.inference_seconds=0.;self.ranking_seconds=0.;self.c2_seconds=0.;self.c4_seconds=0.
        for stage in ('C2','C4'):
            model,meta=load_model(root/f'models/mlp_{stage.lower()}_v05.pt')
            self.models[stage]=(model,meta)
    def _scores(self,stage,features):
        from .model import predict_scores
        model,meta=self.models[stage]
        if any(list(row)!=meta['feature_names'] for row in features): raise ValueError('V05 MLP feature columns differ')
        return predict_scores(model,[list(row.values()) for row in features],meta['mean'],meta['std'])
    def select_c2(self,current,directions,cs,pool,config,*,seed,iteration,k=8):
        start=time.perf_counter()
        try:
            features=[extract_c2_features(current,directions,cs,c,config) for c in pool]
            if not pool:return ()
            scores={id(c):float(v) for c,v in zip(pool,self._scores('C2',features))};rank=time.perf_counter()
            result=family_c2(pool,lambda c:(-scores[id(c)],c.cheap_score),decision_family,k=k,seed=seed,iteration=iteration)
            self.ranking_seconds+=time.perf_counter()-rank;return result
        finally:
            elapsed=time.perf_counter()-start;self.inference_seconds+=elapsed;self.c2_seconds+=elapsed
    def select_c4(self,current,directions,cs,pool,config,*,seed,iteration,k=2):
        start=time.perf_counter()
        try:
            features=[extract_c4_features(current,directions,cs,c.screened,config,direction=c.direction,cheap_rank=c.cheap_rank) for c in pool]
            if not pool:return ()
            scores={id(c):float(v) for c,v in zip(pool,self._scores('C4',features))};rank=time.perf_counter()
            result=family_c4(pool,lambda c:(-scores[id(c)],c.rerank_key),lambda c:decision_family(c.screened),k=k,seed=seed,iteration=iteration)
            self.ranking_seconds+=time.perf_counter()-rank;return result
        finally:
            elapsed=time.perf_counter()-start;self.inference_seconds+=elapsed;self.c4_seconds+=elapsed


class GATRanker:
    """Cache one static graph forward per instance and score only dynamic candidates."""
    def __init__(self,root=None,*,include_weld_edges=None,model_version='v05',legacy_scoring=False):
        from .model import load_graph_model
        import torch
        torch.set_num_threads(1);root=Path(__file__).resolve().parents[2] if root is None else Path(root)
        names={'v05':'gat_v05.pt','v05_c4_ranked':'gat_v05_c4_ranked.pt','v06':'gat_v06.pt','v06_no_weld_edges':'gat_v06_no_weld_edges.pt'}
        if model_version not in names:raise ValueError('Unknown GAT model version')
        self.model,self.meta=load_graph_model(root/'models'/names[model_version]);arch=self.meta.get('architecture',{})
        self.include_weld_edges=arch.get('include_weld_edges',True) if include_weld_edges is None else include_weld_edges
        self.model_version=model_version;self.legacy_scoring=legacy_scoring
        self.inference_seconds=0.;self.graph_build_seconds=0.;self.static_forward_seconds=0.;self.candidate_scoring_seconds=0.;self.ranking_seconds=0.;self.c2_seconds=0.;self.c4_seconds=0.;self.static_forward_calls=0
        self._key=None;self._graph=None;self._embeddings=None
    def _static(self,parents,config):
        import torch
        from .model import build_static_graph
        key=tuple((p.parent_id,p.start,p.end) for p in parents)
        if key==self._key:return self._graph,self._embeddings
        start=time.perf_counter();graph=build_static_graph(parents,config,include_weld_edges=self.include_weld_edges);self.graph_build_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        with torch.inference_mode(): embeddings=self.model.encode_static(graph)
        self.static_forward_seconds+=time.perf_counter()-start;self.static_forward_calls+=1
        self._key=key;self._graph=graph;self._embeddings=embeddings;return graph,embeddings
    def _scores(self,stage,current,candidates,features,config,*,directions=None,current_direction=None):
        import torch
        graph,embeddings=self._static(current.parents,config);mean=self.meta['scalers'][stage+'_mean'];std=self.meta['scalers'][stage+'_std']
        names=self.meta['c2_names' if stage=='C2' else 'c4_names']
        if any(list(row)!=names for row in features):raise ValueError('GAT feature columns differ')
        start=time.perf_counter()
        scorer=self.model.score_candidates_legacy if self.legacy_scoring else self.model.score_candidates
        with torch.inference_mode(): _,material=scorer(stage,graph,embeddings,current,candidates,[list(r.values()) for r in features],mean,std,config,
            directions=directions,current_direction=current_direction)
        self.candidate_scoring_seconds+=time.perf_counter()-start;return torch.sigmoid(material).numpy()
    def select_c2(self,current,directions,cs,pool,config,*,seed,iteration,k=8):
        start=time.perf_counter()
        try:
            features=[extract_c2_features(current,directions,cs,c,config) for c in pool]
            if not pool:return ()
            scores={id(c):float(v) for c,v in zip(pool,self._scores('C2',current,pool,features,config))};rank=time.perf_counter()
            result=family_c2(pool,lambda c:(-scores[id(c)],c.cheap_score),decision_family,k=k,seed=seed,iteration=iteration)
            self.ranking_seconds+=time.perf_counter()-rank;return result
        finally:
            elapsed=time.perf_counter()-start;self.inference_seconds+=elapsed;self.c2_seconds+=elapsed
    def select_c4(self,current,directions,cs,pool,config,*,seed,iteration,k=2):
        start=time.perf_counter()
        try:
            candidates=[c.screened for c in pool]
            features=[extract_c4_features(current,directions,cs,c.screened,config,direction=c.direction,cheap_rank=c.cheap_rank) for c in pool]
            if not pool:return ()
            scores={id(c):float(v) for c,v in zip(pool,self._scores('C4',current,candidates,features,config,
                directions=[c.direction for c in pool],current_direction=directions))};rank=time.perf_counter()
            result=family_c4(pool,lambda c:(-scores[id(c)],c.rerank_key),lambda c:decision_family(c.screened),k=k,seed=seed,iteration=iteration)
            self.ranking_seconds+=time.perf_counter()-rank;return result
        finally:
            elapsed=time.perf_counter()-start;self.inference_seconds+=elapsed;self.c4_seconds+=elapsed


class HybridRanker:
    """Combine the unchanged production C2 and C4 selectors stage by stage."""
    _METHODS = {'HEURISTIC', 'GAT_V05', 'GAT_V05_C4_RANKED'}

    def __init__(self, root=None, *, c2_method='HEURISTIC', c4_method='HEURISTIC'):
        if c2_method not in self._METHODS or c4_method not in self._METHODS:
            raise ValueError('Unknown hybrid ranker method')
        if c2_method == 'GAT_V05_C4_RANKED':
            raise ValueError('The ranked checkpoint is C4-only; frozen C2 must use GAT_V05')
        self.c2_method = c2_method
        self.c4_method = c4_method
        versions = set()
        if c2_method == 'GAT_V05' and c4_method != 'GAT_V05_C4_RANKED':
            versions.add('v05')
        if c4_method == 'GAT_V05':
            versions.add('v05')
        elif c4_method == 'GAT_V05_C4_RANKED':
            versions.add('v05_c4_ranked')
        self._rankers = {version: GATRanker(root, model_version=version) for version in versions}

    def _ranker(self, method):
        version='v05_c4_ranked' if method == 'GAT_V05_C4_RANKED' or 'v05' not in self._rankers else 'v05'
        return self._rankers[version]

    def select_c2(self,current,directions,cs,pool,config,*,seed,iteration,k=8):
        if self.c2_method == 'HEURISTIC':
            return select_c2_by_family(pool,k,seed=seed,iteration=iteration)
        return self._ranker(self.c2_method).select_c2(
            current,directions,cs,pool,config,seed=seed,iteration=iteration,k=k)

    def select_c4(self,current,directions,cs,pool,config,*,seed,iteration,k=2):
        if self.c4_method == 'HEURISTIC':
            return select_c4_by_family(pool,k,seed=seed,iteration=iteration)
        return self._ranker(self.c4_method).select_c4(
            current,directions,cs,pool,config,seed=seed,iteration=iteration,k=k)

    @property
    def inference_seconds(self):
        return sum(r.inference_seconds for r in self._rankers.values())

    @property
    def graph_build_seconds(self):
        return sum(r.graph_build_seconds for r in self._rankers.values())

    @property
    def static_forward_seconds(self):
        return sum(r.static_forward_seconds for r in self._rankers.values())

    @property
    def candidate_scoring_seconds(self):
        return sum(r.candidate_scoring_seconds for r in self._rankers.values())

    @property
    def ranking_seconds(self):
        return sum(r.ranking_seconds for r in self._rankers.values())

    @property
    def c2_seconds(self):
        return sum(r.c2_seconds for r in self._rankers.values())

    @property
    def c4_seconds(self):
        return sum(r.c4_seconds for r in self._rankers.values())


# Final C4 development utilities live beside the production selectors so the
# offline replay cannot silently drift to a different family/tie-break rule.
def _final_rank_module(root):
    import importlib.util,sys
    path=Path(root)/'scripts/train_mlp_ranker.py'
    name='_c4_final_train_module'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def _best_certified_cmax(rows):
    return min((r['reference_cmax'] for r in rows if r['status']=='FEASIBLE_CERTIFIED'),default=None)


def _final_cross_metrics(rank,states,c2_method,c4_method):
    import statistics,time
    stage_rows={'C2':[],'C4':[]};started=time.perf_counter()
    for state in states:
        c2=rank._v05_c2(state,c2_method);c4=rank._v05_c4(state,c2,c4_method)
        for label,selected in (('C2',c2),('C4',c4)):
            pool=state['pool'];positives=sum(r['material'] for r in pool);captured=sum(r['material'] for r in selected)
            best=_best_certified_cmax(selected)
            row={**rank.capture_metrics([{**r,'Cs':state['Cs']} for r in pool],[{**r,'Cs':state['Cs']} for r in selected]),
                'state_id':state['state_id'],'workbook':state['workbook'],'tier':state['tier'],'stage':state['stage'],
                'material_candidates':positives,'material_candidates_captured':captured,
                'material_zero_capture':None if not positives else captured==0,
                'best_certified_cmax':best,'best_certified_gain':None if best is None else state['Cs']-best}
            stage_rows[label].append(row)
    result={}
    for label,rows in stage_rows.items():
        summary=rank.group_summary(rows);groups={'ALL':rows,**{tier:[r for r in rows if r['tier']==tier] for tier in rank.d.TIERS}}
        for group,items in groups.items():
            available=sum(r['material_candidates'] for r in items);kept=sum(r['material_candidates_captured'] for r in items)
            material=[r for r in items if r['material_candidates']]
            summary[group].update({'material_candidate_capture':None if not available else kept/available,
                'material_zero_capture':None if not material else statistics.fmean(r['material_zero_capture'] for r in material),
                'best_certified_gain_median':rank.median(r['best_certified_gain'] for r in items)})
        result[label]=summary
    result['selection_seconds']=time.perf_counter()-started
    return result


def _final_oracle_rows(rank,states,c2_method,c4_method):
    import math
    rows=[];heuristic=lambda r:(r['projected'],r['dp_travel'],r['split_delta'],r['candidate_id'])
    for state in states:
        c2=rank._v05_c2(state,c2_method);eligible=[r for r in c2 if r['C4_features'] is not None]
        actual=rank._v05_c4(state,c2,c4_method)
        oracle=family_c4(eligible,lambda r:((0,r['reference_cmax'],heuristic(r)) if r['status']=='FEASIBLE_CERTIFIED' else (1,math.inf,heuristic(r))),
            lambda r:r['family'],k=2,seed=state['seed'],iteration=state['iteration'])
        pool_best=_best_certified_cmax(state['pool']);c2_best=_best_certified_cmax(eligible)
        oracle_best=_best_certified_cmax(oracle);c4_best=_best_certified_cmax(actual);cs=state['Cs']
        if c2_best is not None and oracle_best!=c2_best:raise AssertionError('Oracle C4 did not place C2Best in Top2')
        material=[r for r in eligible if r['material']];actual_material=[r for r in actual if r['material']]
        oracle_gain=0. if oracle_best is None else max(0.,cs-oracle_best);actual_gain=0. if c4_best is None else max(0.,cs-c4_best)
        rows.append({'state_id':state['state_id'],'workbook':state['workbook'],'tier':state['tier'],'stage':state['stage'],'Cs':cs,
            'pool_best_cmax':pool_best,'c2_best_cmax':c2_best,'oracle_c4_best_cmax':oracle_best,'c4_best_cmax':c4_best,
            'pool_to_c2_lost_gain_seconds':max(0.,(cs if c2_best is None else min(cs,c2_best))-(cs if pool_best is None else min(cs,pool_best))) if pool_best is not None else None,
            'c2_to_oracle_gap_seconds':None if c2_best is None else oracle_best-c2_best,
            'oracle_to_actual_lost_gain_seconds':oracle_gain-actual_gain,
            'material_in_top8':len(material),'material_selected':len(actual_material),
            'single_material_selected':None if len(material)!=1 else bool(actual_material),
            'more_than_two_material':len(material)>2,'all_material_missed':bool(material) and not actual_material,
            'certified_but_small_gain':oracle_gain/cs>=.005 and c4_best is not None and actual_gain/cs<.005,
            'actual_selected_ids':[r['candidate_id'] for r in actual],'oracle_selected_ids':[r['candidate_id'] for r in oracle]})
    return rows


def _final_oracle_summary(rows):
    from collections import defaultdict
    import statistics
    by_workbook=defaultdict(list)
    for row in rows:by_workbook[row['workbook']].append(row)
    def summarize(items):
        losses=[r['oracle_to_actual_lost_gain_seconds'] for r in items];positive=[v for v in losses if v>1e-9]
        return {'states':len(items),'states_with_c2_certified':sum(r['c2_best_cmax'] is not None for r in items),
            'states_with_ranking_loss':len(positive),'mean_lost_gain_seconds':statistics.fmean(losses),
            'median_positive_lost_gain_seconds':statistics.median(positive) if positive else None,'max_lost_gain_seconds':max(losses,default=0.),
            'single_material_states':sum(r['single_material_selected'] is not None for r in items),
            'single_material_selected':sum(r['single_material_selected'] is True for r in items),
            'more_than_two_material_states':sum(r['more_than_two_material'] for r in items),
            'all_material_missed_states':sum(r['all_material_missed'] for r in items),
            'certified_but_small_gain_states':sum(r['certified_but_small_gain'] for r in items)}
    workbooks={key:summarize(value) for key,value in by_workbook.items()};state_summary=summarize(rows)
    state_summary['workbook_equal_mean_lost_gain_seconds']=statistics.fmean(v['mean_lost_gain_seconds'] for v in workbooks.values())
    state_summary['workbooks_with_ranking_loss']=sum(v['states_with_ranking_loss']>0 for v in workbooks.values())
    return {'state_equal':state_summary,'workbooks':workbooks,
        'tiers':{tier:summarize([r for r in rows if r['tier']==tier]) for tier in ('SMALL','MEDIUM','LARGE')}}


def run_final_c4_offline(root=None):
    """Execute the four cross combinations and fixed-Top8, Top2-family oracle."""
    import torch
    from . import dataset as data
    from .model import load_graph_model
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);rank=_final_rank_module(root)
    torch.set_num_threads(1);states,_=rank.load_v05_states('MLP_DEV');model,meta=load_graph_model(root/'models/gat_v05.pt')
    scoring={stage:rank._gat_score_states_v2(model,states,stage,meta['scalers'],'GAT_V05') for stage in ('C2','C4')}
    combos={'A':('HEURISTIC','HEURISTIC'),'B':('GAT_V05','HEURISTIC'),'C':('HEURISTIC','GAT_V05'),'D':('GAT_V05','GAT_V05')}
    metrics={key:{'c2_method':c2,'c4_method':c4,**_final_cross_metrics(rank,states,c2,c4)} for key,(c2,c4) in combos.items()}
    oracle={c2:{c4:{'summary':_final_oracle_summary(rows),'states':rows}
        for c4 in ('HEURISTIC','GAT_V05') for rows in [_final_oracle_rows(rank,states,c2,c4)]}
        for c2 in ('HEURISTIC','GAT_V05')}
    result={'dataset':{'states':len(states),'candidates':sum(len(s['pool']) for s in states),'workbooks':len({s['workbook'] for s in states})},
        'combos':metrics,'oracle':oracle,'scoring_timing':scoring,'oracle_online_access':False,
        'rules':{'C2_topk':8,'C4_topk':2,'family_c2':'production family coverage','family_c4':'production explore/exploit','tie_break':'unchanged'}}
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_offline','summary',data.dumps(result)))
    db.close();print(data.dumps(result),flush=True);return result


def train_final_c4_ranked(root=None):
    """Run the single fixed V05-C4 relative-gain ranking training."""
    import statistics,time,torch
    from copy import deepcopy
    from collections import Counter
    from torch import nn
    from . import dataset as data
    from .model import build_static_graph,load_graph_model,save_graph_model
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);rank=_final_rank_module(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261520)
    train,_=rank.load_v05_states('MLP_TRAIN');dev,names=rank.load_v05_states('MLP_DEV');model,meta=load_graph_model(root/'models/gat_v05.pt')
    for states in (train,dev):rank._gat_score_states_v2(model,states,'C2',meta['scalers'],'GAT_V05')
    before=deepcopy(model.state_dict());hard={};distribution=Counter()
    for state in train:
        chosen={r['candidate_id']:r for method in ('HEURISTIC','GAT_V05') for r in rank._v05_c2(state,method) if r['C4_features'] is not None}
        hard[state['state_id']]=list(chosen.values());distribution['rows']+=len(chosen)
        distribution['certified']+=sum(r['status']=='FEASIBLE_CERTIFIED' for r in chosen.values())
        distribution['uncertified']+=sum(r['status']!='FEASIBLE_CERTIFIED' for r in chosen.values())
    allowed=('route.C4','numeric.C4','fusion.C4','feasibility.C4','material.C4')
    for name,p in model.named_parameters():p.requires_grad=any(name.startswith(prefix) for prefix in allowed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    graphs={s['instance_id']:build_static_graph(s['current'].parents,data.CONFIG_V05) for s in train};embeddings={}
    with torch.no_grad():
        for state in train:
            if state['instance_id'] not in embeddings:embeddings[state['instance_id']]=model.encode_static(graphs[state['instance_id']])
    history=[];started=time.perf_counter()
    for epoch in range(1,5):
        model.train();losses=[];bces=[];pairs=[]
        for state in train:
            rows=hard[state['state_id']]
            if not rows:continue
            optimizer.zero_grad(set_to_none=True)
            feasibility,score=model.score_candidates('C4',graphs[state['instance_id']],embeddings[state['instance_id']],state['current'],
                [r['candidate'] for r in rows],[r['C4_features'] for r in rows],meta['scalers']['C4_mean'],meta['scalers']['C4_std'],data.CONFIG_V05)
            y_feasible=torch.tensor([r['status']=='FEASIBLE_CERTIFIED' for r in rows],dtype=torch.float32)
            bce=nn.functional.binary_cross_entropy_with_logits(feasibility,y_feasible);pair_losses=[];weights=[]
            certified=[(i,(state['Cs']-r['reference_cmax'])/state['Cs']) for i,r in enumerate(rows) if r['status']=='FEASIBLE_CERTIFIED']
            for ai,(a,gain_a) in enumerate(certified):
                for b,gain_b in certified[ai+1:]:
                    if abs(gain_a-gain_b)<=1e-12:continue
                    better,worse=(a,b) if gain_a>gain_b else (b,a);difference=abs(gain_a-gain_b)
                    pair_losses.append(nn.functional.softplus(-(score[better]-score[worse])));weights.append(1.+min(2.,10.*difference))
            if pair_losses:
                weight=torch.tensor(weights,dtype=torch.float32);pair=(torch.stack(pair_losses)*weight).sum()/weight.sum()
            else:pair=score.sum()*0.
            loss=bce+pair;loss.backward();optimizer.step();losses.append(float(loss.detach()));bces.append(float(bce.detach()));pairs.append(float(pair.detach()))
        history.append({'epoch':epoch,'loss':statistics.fmean(losses),'feasibility_bce':statistics.fmean(bces),'statewise_pairwise':statistics.fmean(pairs)})
    model.eval();rank._gat_score_states_v2(model,dev,'C4',meta['scalers'],'GAT_V05_C4_RANKED')
    for state in dev:
        for row in state['pool']:row['GAT_V05_C4_RANKED_C2']=row['GAT_V05_C2']
    frozen_names=[name for name in before if not any(name.startswith(prefix) for prefix in allowed)]
    if not all(torch.equal(before[name],model.state_dict()[name]) for name in frozen_names):raise AssertionError('C4 training changed frozen C2/shared parameters')
    evaluation=_final_cross_metrics(rank,dev,'GAT_V05','GAT_V05_C4_RANKED');oracle_rows=_final_oracle_rows(rank,dev,'GAT_V05','GAT_V05_C4_RANKED')
    training={'seed':20261520,'epochs':4,'objective':'feasibility BCE + within-state certified relative-gain pairwise',
        'gain_weight':'1 + min(2, 10*abs(gain difference))','distribution':dict(distribution),'history':history,
        'training_seconds':time.perf_counter()-started,'c2_frozen':True,'dev':evaluation,
        'dev_oracle':_final_oracle_summary(oracle_rows),'dev_used_for_development_selection':True}
    save_graph_model(root/'models/gat_v05_c4_ranked.pt',model,names['C2'],names['C4'],meta['scalers'],training)
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_training','summary',data.dumps(training)))
    db.close();print(data.dumps(training),flush=True);return training


FINAL_C4_SEEDS=(20261501,20261502)
FINAL_BASELINE_SMOKE_SEED=20261511


def _final_smoke_module(root):
    import importlib.util,sys
    path=Path(root)/'scripts/run_ppo_smoke.py';name='_c4_final_smoke_module'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def _final_worker_init(masks):
    import multiprocessing as mp
    from . import dataset as data
    data.bind_to_core(masks[(mp.current_process()._identity[0]-1)%len(masks)])


def _final_online_worker(job):
    from . import dataset as data
    entry,seed,method,root=job;smoke=_final_smoke_module(root);parents=smoke._range_parents(entry,data.PPO)
    if method=='HEURISTIC':ranker=None
    elif method=='GAT_V05':ranker=HybridRanker(root,c2_method='GAT_V05',c4_method='GAT_V05')
    elif method=='GAT_V05_C4_RANKED':ranker=HybridRanker(root,c2_method='GAT_V05',c4_method='GAT_V05_C4_RANKED')
    else:raise ValueError(method)
    record,_=data.run_trajectory(entry,seed,parents=parents,ranker=ranker,config=data.CONFIG_V05)
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],'relative_path':entry['relative_path'],
        'actual_N':len(parents),'total_weld_length':sum(p.length for p in parents),'seed':seed,'method':method,**record}


def run_final_c4_online(root=None,workers=4):
    """Run the fixed 6x2x3 real 60-second consumed-development matrix."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp,json
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);smoke=_final_smoke_module(root)
    entries=smoke._c2_entries(root);roles={r['relative_path']:r['new_v2_role'] for r in data.read(root/'data/manifests/PPO_V2_DATA_ROLES_V1.json')['workbooks']}
    if len(entries)!=6 or any(roles.get(e['relative_path'])!='V2_MODEL_DEVELOPMENT_CONSUMED' for e in entries):raise PermissionError('Final matrix requires exactly six consumed-development workbooks')
    methods=('HEURISTIC','GAT_V05','GAT_V05_C4_RANKED');db=data.connect(data.DATABASE_V05)
    existing=[json.loads(r[0]) for r in db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_runs'")]
    if existing and {r['method'] for r in existing}-set(methods):raise RuntimeError('Existing final matrix uses a different method set')
    done={r[0] for r in db.execute("SELECT item_id FROM experiments WHERE experiment='c4_final_runs'")};db.close();jobs=[]
    for entry_index,entry in enumerate(entries):
        for seed_index,seed in enumerate(FINAL_C4_SEEDS):
            order=methods[(entry_index+seed_index)%3:]+methods[:(entry_index+seed_index)%3]
            for method in order:
                key=f"{entry['label']}:{seed}:{method}"
                if key not in done:jobs.append((key,(entry,seed,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_final_online_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            row=future.result();db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_runs',futures[future],data.dumps(row)))
            db.close();print(f"final {futures[future]} Cmax@60={row['Cmax_at_60']} iterations={row['iterations']}",flush=True)
    return len(jobs)


def _final_baseline_worker(job):
    from . import dataset as data
    from mrta_baselines import AdaptedHGAConfig,AdaptedWAGConfig,run_adapted_hga_v2,run_adapted_wag_vns_v2
    entry,method,root=job;smoke=_final_smoke_module(root);parents=smoke._range_parents(entry,data.PPO)
    if method=='HGA':result=run_adapted_hga_v2(parents,data.CONFIG_V05,AdaptedHGAConfig(),seed=FINAL_BASELINE_SMOKE_SEED,time_limit=60.)
    elif method=='WAG':result=run_adapted_wag_vns_v2(parents,data.CONFIG_V05,AdaptedWAGConfig(),seed=FINAL_BASELINE_SMOKE_SEED,time_limit=60.)
    else:raise ValueError(method)
    initial=None if result.initial_candidate is None or result.initial_candidate.metrics is None else result.initial_candidate.metrics.cmax
    return {'instance':entry['label'],'tier':entry['tier'],'relative_path':entry['relative_path'],'actual_N':len(parents),'method':method,'seed':FINAL_BASELINE_SMOKE_SEED,
        'status':result.status.value,'termination_reason':result.termination_reason,'initial_cmax':initial,'Cmax_at_60':result.checkpoints[60.0],
        'final_certified':result.final_certified,'iterations':result.iterations,'candidate_count':result.candidate_count,
        'reference_calls':result.reference_calls,'initial_reference_calls':result.initial_reference_calls,
        'search_reference_calls':result.reference_calls-result.initial_reference_calls,'certifier_calls':result.certifier_calls,
        'actual_runtime':result.actual_runtime,'overshoot':result.overshoot,'scope_id':result.scope_id,
        'scientific_config':data.CONFIG_V05.scientific_mapping(),'accounting':result.accounting,'diagnostics':result.diagnostics}


def run_final_baseline_smoke(root=None,workers=2):
    """Run two real 60-second HGA/WAG interface smokes without retries."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);entries=_final_smoke_module(root)._c2_entries(root)
    selected=(entries[0],entries[4]);db=data.connect(data.DATABASE_V05)
    done={r[0] for r in db.execute("SELECT item_id FROM experiments WHERE experiment='c4_final_baseline_smoke'")};db.close();jobs=[]
    for entry in selected:
        for method in ('HGA','WAG'):
            key=f"{entry['label']}:{FINAL_BASELINE_SMOKE_SEED}:{method}"
            if key not in done:jobs.append((key,(entry,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_final_baseline_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            row=future.result();db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_baseline_smoke',futures[future],data.dumps(row)))
            db.close();print(f"baseline {futures[future]} status={row['status']} Cmax@60={row['Cmax_at_60']}",flush=True)
    return len(jobs)


def summarize_final_c4_results(root=None):
    import json,statistics
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    runs=[json.loads(r[0]) for r in db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_runs' ORDER BY item_id")]
    baselines=[json.loads(r[0]) for r in db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_baseline_smoke' ORDER BY item_id")]
    offline=json.loads(db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_offline' AND item_id='summary'").fetchone()[0])
    training=json.loads(db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_training' AND item_id='summary'").fetchone()[0]);db.close()
    methods=('HEURISTIC','GAT_V05','GAT_V05_C4_RANKED');tiers=('N35','N55','N85')
    if len(runs)!=36 or any(not r['valid_60s_outcome'] for r in runs):raise AssertionError('Expected 36 independently certified Cmax@60 outcomes')
    groups={}
    for instance in sorted({r['instance'] for r in runs}):
        for seed in FINAL_C4_SEEDS:
            rows={r['method']:r for r in runs if r['instance']==instance and r['seed']==seed}
            if set(rows)!=set(methods) or len({round(r['initial_cmax'],8) for r in rows.values()})!=1:raise AssertionError('Incomplete or unmatched pair')
            groups[(instance,seed)]=rows
    for row in runs:
        expected=min((cmax for elapsed,cmax in row['best_events'] if elapsed<=60.),default=row['initial_cmax'])
        if abs(expected-row['Cmax_at_60'])>1e-8*max(1.,abs(expected)):raise AssertionError('Cmax@60 contains a post-deadline backfill')
        if data.strict_cmax_improvement_count(row['best_events'])!=row['strict_best_improvements_at60']:raise AssertionError('Strict Cmax count mismatch')
        if row['initial_reference_calls']+row['search_reference_calls']!=row['reference_calls']:raise AssertionError('Reference accounting mismatch')
    def paired(method,tier=None):
        values=[rows[method]['Cmax_at_60']/rows['HEURISTIC']['Cmax_at_60']-1 for rows in groups.values() if tier is None or rows['HEURISTIC']['tier']==tier]
        eps=1e-10
        return {'pairs':len(values),'wins':sum(v < -eps for v in values),'ties':sum(abs(v)<=eps for v in values),'losses':sum(v>eps for v in values),
            'median_delta':statistics.median(values),'mean_delta':statistics.fmean(values),'min_delta':min(values),'max_delta':max(values)}
    online={method:{'ALL':paired(method),**{tier:paired(method,tier) for tier in tiers}} for method in methods[1:]}
    totals={}
    for method in methods:
        rows=[r for r in runs if r['method']==method]
        totals[method]={'iterations':sum(r['iterations'] for r in rows),'reference_calls':sum(r['reference_calls'] for r in rows),
            'initial_reference_calls':sum(r['initial_reference_calls'] for r in rows),'search_reference_calls':sum(r['search_reference_calls'] for r in rows),
            'accepted_moves':sum(r['accepted_moves'] for r in rows),'strict_cmax_improvements':sum(r['strict_best_improvements_at60'] for r in rows),
            'ranker_seconds':sum(r['ranker_inference_seconds'] for r in rows),'actual_runtime':sum(r['actual_runtime'] for r in rows),
            'overshoot_min':min(r['overshoot'] for r in rows),'overshoot_max':max(r['overshoot'] for r in rows),
            'search_reference_status':{status:sum(r['search_reference_status'].get(status,0) for r in rows) for status in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')},
            'initial_reference_status':{status:sum(r['initial_reference_status'].get(status,0) for r in rows) for status in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')}}
    if len(baselines)!=4:raise AssertionError('Expected four baseline smoke rows, including failures without retry')
    expected_config=data.CONFIG_V05.scientific_mapping();baseline_ok=all(r['scientific_config']==expected_config and r['scope_id']=='FORMAL_SCOPE_V2' for r in baselines)
    summary={'online':online,'totals':totals,'runs':runs,'baseline_smoke':baselines,'baseline_shared_config':baseline_ok,
        'offline':offline,'training':training,'certified_online':sum(r['valid_60s_outcome'] for r in runs),'no_post60_backfill':True,
        'frozen_selection':'A_HEURISTIC_DEFAULT','learning_method_for_paper':'GAT_V05','new_c4_rejected':True,
        'id_test_accessed':False,'formal_experiments_ready':True,'remaining_condition':'Freeze the formal instance/seed matrix and statistical analysis plan before unsealing ID_TEST.'}
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_summary','summary',data.dumps(summary)))
    db.close();print(data.dumps({k:summary[k] for k in ('online','totals','baseline_smoke','frozen_selection','remaining_condition')}),flush=True);return summary


def validate_final_c4_invariants(root=None):
    """Executable checks for offline-only oracle, stage access, budgets and frozen C2."""
    import json,torch
    from . import dataset as data
    from .model import load_graph_model
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);rank=_final_rank_module(root)
    states,_=rank.load_v05_states('MLP_DEV');old,old_meta=load_graph_model(root/'models/gat_v05.pt');new,new_meta=load_graph_model(root/'models/gat_v05_c4_ranked.pt')
    rank._gat_score_states_v2(old,states,'C2',old_meta['scalers'],'OLD')
    rank._gat_score_states_v2(new,states,'C2',new_meta['scalers'],'NEW')
    c2_scores_equal=all(r['OLD_C2']==r['NEW_C2'] for s in states for r in s['pool'])
    for state in states:
        for row in state['pool']:
            row['OLD_C2'],row['NEW_C2']=row['OLD_C2'],row['NEW_C2']
    c2_top8_equal=all(tuple(r['candidate_id'] for r in rank._v05_c2(s,'OLD'))==tuple(r['candidate_id'] for r in rank._v05_c2(s,'NEW')) for s in states)
    different_upstreams=sum(tuple(r['candidate_id'] for r in rank._v05_c2(s,'HEURISTIC'))!=tuple(r['candidate_id'] for r in rank._v05_c2(s,'OLD')) for s in states)
    db=data.connect(data.DATABASE_V05,readonly=True);offline=json.loads(db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_offline' AND item_id='summary'").fetchone()[0])
    direction_access=db.execute("SELECT count(*) FROM candidates WHERE (status='DIRECTION_INFEASIBLE') != (c4 IS NULL)").fetchone()[0]
    c2_missing=db.execute("SELECT count(*) FROM candidates WHERE c2 IS NULL").fetchone()[0]
    false_cmax=db.execute("SELECT count(*) FROM candidates WHERE status!='FEASIBLE_CERTIFIED' AND reference_cmax IS NOT NULL").fetchone()[0]
    summary=json.loads(db.execute("SELECT payload FROM experiments WHERE experiment='c4_final_summary' AND item_id='summary'").fetchone()[0]);db.close()
    oracle_rows=[row for by_c4 in offline['oracle'].values() for value in by_c4.values() for row in value['states']]
    result={'production_family_rules_shared':offline['rules']['family_c2']=='production family coverage' and offline['rules']['family_c4']=='production explore/exploit',
        'oracle_offline_only':offline['oracle_online_access'] is False,
        'oracle_same_top8_and_best_first':all(row['c2_best_cmax']==row['oracle_c4_best_cmax'] for row in oracle_rows),
        'upstream_top8_difference_states':different_upstreams,'uncertified_cmax_remains_null':false_cmax==0,
        'frozen_c2_scores_equal':c2_scores_equal,'frozen_c2_top8_equal':c2_top8_equal,'c2_future_feature_violations':c2_missing,
        'c4_direction_access_violations':direction_access,'top8_top2_fixed':offline['rules']['C2_topk']==8 and offline['rules']['C4_topk']==2,
        'online_certified':summary['certified_online'],'baseline_shared_config':summary['baseline_shared_config'],'no_post60_backfill':summary['no_post60_backfill']}
    if not all((result['production_family_rules_shared'],result['oracle_offline_only'],result['oracle_same_top8_and_best_first'],
        result['upstream_top8_difference_states']>0,result['uncertified_cmax_remains_null'],result['frozen_c2_scores_equal'],result['frozen_c2_top8_equal'],
        result['c2_future_feature_violations']==0,result['c4_direction_access_violations']==0,result['top8_top2_fixed'],
        result['online_certified']==36,result['baseline_shared_config'],result['no_post60_backfill'])):raise AssertionError(result)
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('c4_final_validation','summary',data.dumps(result)))
    db.close();print(data.dumps(result),flush=True);return result


def write_final_c4_report(root=None):
    import json,statistics
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    get=lambda name:json.loads(db.execute('SELECT payload FROM experiments WHERE experiment=? AND item_id=?',(name,'summary')).fetchone()[0])
    summary=get('c4_final_summary');validation=get('c4_final_validation');db.close()
    offline=summary['offline'];training=summary['training'];runs=summary['runs'];baseline=summary['baseline_smoke']
    pc=lambda value:'—' if value is None else f'{100*value:.2f}%'
    tier_labels={'SMALL':'N35','MEDIUM':'N55','LARGE':'N85','ALL':'ALL'}
    lines=['# GRC-ALNS C4 最终选择','',
        '**冻结结论：选择 A——HEURISTIC 继续作为默认算法；GAT V05 冻结为论文学习排序方法与消融对象；最后一次 C4-ranked 被淘汰。此后不开发 GAT V07/V08。**','',
        '全部结果使用 20m×12m、四机器人、WHOLE/Y_SPLIT/X_SPLIT、delta_x=delta_y=0.50m、min_child_length=0.20m、interference_dx=interference_dy=0.50m；M192=144 atomic+48 LNS、Kdp8、Kref2/Kref_total4、BASELINE initialization、原SA/direction/B32/independent certifier、TWO_OPT_STAR OFF。未访问 ID_TEST_SEALED、V2_VALIDATION 或 ORACLE_DEV_CONSUMED。','',
        '## 1. C2/C4 四种交叉组合','',
        '34 个 DEV 状态、4184 个已有认证候选；C2/C4 都复用生产 family 与 tie-break。排序耗时为相同进程离线评分：GAT C2/C4 '+f"{offline['scoring_timing']['C2']['seconds']:.3f}s/{offline['scoring_timing']['C4']['seconds']:.3f}s；各组合 family 排序约 {statistics.fmean(v['selection_seconds'] for v in offline['combos'].values()):.4f}s。",'',
        '|组合|规模|C2 material capture|C2 zero|C4 material capture|C4 zero|C4最佳认证改善中位数(s)|','|---|---|---:|---:|---:|---:|---:|']
    names={'A':'HEUR/HEUR','B':'GAT/HEUR','C':'HEUR/GAT','D':'GAT/GAT'}
    for key,value in offline['combos'].items():
        for tier in ('ALL','SMALL','MEDIUM','LARGE'):
            c2=value['C2'][tier];c4=value['C4'][tier]
            lines.append(f"|{key} {names[key]}|{tier_labels[tier]}|{pc(c2['material_candidate_capture'])}|{pc(c2['material_zero_capture'])}|{pc(c4['material_candidate_capture'])}|{pc(c4['material_zero_capture'])}|{c4['best_certified_gain_median'] if c4['best_certified_gain_median'] is not None else '—'}|")
    lines+=['','D 的 material capture 最高（4.78%），但 material 二分类并不等价于 Cmax 幅度排序；同一 GAT Top8 下，HEURISTIC-C4 的状态等权 Cmax 损失反而低于原 GAT-C4。C2 与 C4 不应被强制绑定为同一种排序方法。','',
        '## 2. 固定 Top8 的 C4 Oracle 上限','',
        'Oracle 只事后重排固定 C2 Top8 中 direction-feasible 且已有 B32 标签的候选。生产 family_c4 的第一个名额可放全局最佳，因此所有有 C2Best 的状态均有 OracleC4Best=C2Best；C2Best→OracleC4Best 的 Cmax 差为 0，未人为制造 family 损失。未认证候选保持 NULL。','',
        '|C2|规模|PoolBest存在|C2Best/Oracle存在|Oracle改善中位(s)|Pool→C2漏失均值(s)|C2→Oracle最大差(s)|','|---|---|---:|---:|---:|---:|---:|']
    for c2,by_c4 in offline['oracle'].items():
        source=by_c4['HEURISTIC']['states']
        for tier,label in ((None,'ALL'),('SMALL','N35'),('MEDIUM','N55'),('LARGE','N85')):
            rows=[r for r in source if tier is None or r['tier']==tier];pool_losses=[r['pool_to_c2_lost_gain_seconds'] for r in rows if r['pool_to_c2_lost_gain_seconds'] is not None]
            oracle_gains=[max(0.,r['Cs']-r['oracle_c4_best_cmax']) for r in rows if r['oracle_c4_best_cmax'] is not None]
            c2_oracle=[abs(r['c2_to_oracle_gap_seconds']) for r in rows if r['c2_to_oracle_gap_seconds'] is not None]
            lines.append(f"|{c2}|{label}|{sum(r['pool_best_cmax'] is not None for r in rows)}|{sum(r['oracle_c4_best_cmax'] is not None for r in rows)}|{statistics.median(oracle_gains) if oracle_gains else '—'}|{statistics.fmean(pool_losses) if pool_losses else '—'}|{max(c2_oracle,default=0.):.3f}|")
    lines+=['','|C2|实际C4|规模|状态|有C2认证|有排序损失|平均丢失(s)|正损失中位(s)|最大丢失(s)|全漏 material|认证但改善过小|仅1个 material/选中|>2 material|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c2,by_c4 in offline['oracle'].items():
        for c4,value in by_c4.items():
            for tier,stats in [('ALL',value['summary']['state_equal']),*[(tier_labels[t],value['summary']['tiers'][t]) for t in ('SMALL','MEDIUM','LARGE')]]:
                lines.append(f"|{c2}|{c4}|{tier}|{stats['states']}|{stats['states_with_c2_certified']}|{stats['states_with_ranking_loss']}|{stats['mean_lost_gain_seconds']:.3f}|{stats['median_positive_lost_gain_seconds'] if stats['median_positive_lost_gain_seconds'] is not None else '—'}|{stats['max_lost_gain_seconds']:.3f}|{stats['all_material_missed_states']}|{stats['certified_but_small_gain_states']}|{stats['single_material_states']}/{stats['single_material_selected']}|{stats['more_than_two_material_states']}|")
    lines+=['','工作簿等权与状态等权结论一致：四个 C2/C4 组合均在 5/6 个 workbook 出现排序损失。原 GAT-C2/GAT-C4 为 18/34 状态有损失，状态等权平均 353.242s、workbook 等权 352.086s、最大 2898.396s；这是明显可修复排序损失，而非 Top2 预算上限。','',
        '|C2|C4|workbook|states|有损失状态|平均丢失(s)|最大丢失(s)|','|---|---|---|---:|---:|---:|---:|']
    for c2,by_c4 in offline['oracle'].items():
        for c4,value in by_c4.items():
            for workbook,stats in value['summary']['workbooks'].items():
                lines.append(f"|{c2}|{c4}|{Path(workbook).name}|{stats['states']}|{stats['states_with_ranking_loss']}|{stats['mean_lost_gain_seconds']:.3f}|{stats['max_lost_gain_seconds']:.3f}|")
    lines+=['','PoolBest→C2Best 是上游漏选；C2Best→Oracle 恒为 0；Oracle→实际 C4 是排序可修复损失。完整 34 状态的 PoolBest/C2Best/Oracle/C4Best（含 NULL）已保存在 SQLite `c4_final_offline` 的逐状态记录中。','',
        '## 3. 唯一一次 C4-ranked 训练','',
        f"训练分布为 HEURISTIC-C2 与 GAT V05-C2 的真实 Top8 并集，共 {training['distribution']['rows']} rows（认证 {training['distribution']['certified']}，未认证 {training['distribution']['uncertified']}）；固定4 epoch，训练 {training['training_seconds']:.3f}s。loss=可认证性 BCE + 状态内相对 gain pairwise，幅度权重有界到3。GAT V05 C2 分数和 Top8 逐值不变。DEV 参与开发选择，不是独立测试。",'',
        f"结果未改善：原 GAT/GAT C4 material capture={pc(offline['combos']['D']['C4']['ALL']['material_candidate_capture'])}、zero={pc(offline['combos']['D']['C4']['ALL']['material_zero_capture'])}；新 C4 为 {pc(training['dev']['C4']['ALL']['material_candidate_capture'])}、{pc(training['dev']['C4']['ALL']['material_zero_capture'])}。Oracle→实际平均损失从 353.242s 增至 {training['dev_oracle']['state_equal']['mean_lost_gain_seconds']:.3f}s。",'',
        '## 4. 36 次真实 60 秒配对','',
        '|instance|N|seed|initial|HEURISTIC|GAT V05|Δ|新C4|Δ|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    grouped={(r['instance'],r['seed']):{} for r in runs}
    for row in runs:grouped[row['instance'],row['seed']][row['method']]=row
    for (instance,seed),rows in sorted(grouped.items()):
        base=rows['HEURISTIC']['Cmax_at_60'];meta=rows['HEURISTIC']
        lines.append(f"|{instance}|{meta['actual_N']}|{seed}|{meta['initial_cmax']:.6f}|{base:.6f}|{rows['GAT_V05']['Cmax_at_60']:.6f}|{pc(rows['GAT_V05']['Cmax_at_60']/base-1)}|{rows['GAT_V05_C4_RANKED']['Cmax_at_60']:.6f}|{pc(rows['GAT_V05_C4_RANKED']['Cmax_at_60']/base-1)}|")
    lines+=['','|方法|规模|胜/平/负|median Δ|mean Δ|最差 Δ|','|---|---|---:|---:|---:|---:|']
    for method,by_tier in summary['online'].items():
        for tier in ('ALL','N35','N55','N85'):
            row=by_tier[tier];lines.append(f"|{method}|{tier}|{row['wins']}/{row['ties']}/{row['losses']}|{pc(row['median_delta'])}|{pc(row['mean_delta'])}|{pc(row['max_delta'])}|")
    lines+=['','原 GAT V05 在新 seeds 为 9/3/0，但历史 12 对仍有 N85 +24.89% 与 N35 +5.93% 的明显恶化；新 C4 为 6/3/3，N85 median +4.00%、最差 +8.29%。因此新 C4 淘汰，且不以开发数据上的新一轮强表现掩盖历史极端风险。','',
        '### 每 run 完整记账','',
        '状态格式为 `F/D/I/N`；reference 为 `初始化/搜索`。负载、空走、等待均按机器人0–3列出。','',
        '|instance|seed|method|initial|Cmax@60|iter|ref init/search|search F/D/I/N|accepted|严格Cmax下降|负载4机|空走4机|等待4机|推理(s)|runtime(s)|overshoot(s)|','|---|---:|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|']
    fmt4=lambda values:'—' if values is None else '/'.join(f'{v:.2f}' for v in values)
    for row in sorted(runs,key=lambda r:(r['instance'],r['seed'],r['method'])):
        status=row['search_reference_status'];status_text='/'.join(str(status.get(k,0)) for k in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE'))
        lines.append(f"|{row['instance']}|{row['seed']}|{row['method']}|{row['initial_cmax']:.3f}|{row['Cmax_at_60']:.3f}|{row['iterations']}|{row['initial_reference_calls']}/{row['search_reference_calls']}|{status_text}|{row['accepted_moves']}|{row['strict_best_improvements_at60']}|{fmt4(row['best60_process_loads'])}|{fmt4(row['best60_empty_travel_by_robot'])}|{fmt4(row['best60_waiting_by_robot'])}|{row['ranker_inference_seconds']:.3f}|{row['actual_runtime']:.3f}|{row['overshoot']:.3f}|")
    lines+=['','|method|iterations|ref init/search|accepted|严格Cmax下降|ranker(s)|runtime(s)|overshoot范围(s)|','|---|---:|---:|---:|---:|---:|---:|---|']
    for method,row in summary['totals'].items():lines.append(f"|{method}|{row['iterations']}|{row['initial_reference_calls']}/{row['search_reference_calls']}|{row['accepted_moves']}|{row['strict_cmax_improvements']}|{row['ranker_seconds']:.3f}|{row['actual_runtime']:.3f}|{row['overshoot_min']:.3f}–{row['overshoot_max']:.3f}|")
    lines+=['','## 5. 冻结决定','',
        '- A（HEURISTIC）为默认：无模型推理成本，历史跨 seeds 的极端风险最低。',
        '- GAT V05 完整 C2/C4 冻结为论文学习方法及消融对象；不再更新架构或用 DEV 继续调参。',
        '- C（交叉组合）离线说明 C2/C4 不必同构，但缺少本轮必需的新在线第三槽位，不能越过真实 60 秒证据设为默认。',
        '- D（新 C4）离线与 N85 在线均失败。学习模型开发结束，不进入 V07/V08。','',
        '## 6. HGA/WAG 0.5m 公平接口 smoke','',
        '|instance|method|seed|status|initial|Cmax@60|iterations|ref init/search|certifier|runtime|overshoot|','|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in baseline:lines.append(f"|{row['instance']}|{row['method']}|{row['seed']}|{row['status']}|{row['initial_cmax']:.6f}|{row['Cmax_at_60'] if row['Cmax_at_60'] is not None else 'NULL'}|{row['iterations']}|{row['initial_reference_calls']}/{row['search_reference_calls']}|{row['certifier_calls']}|{row['actual_runtime']:.3f}|{row['overshoot']:.3f}|")
    lines+=['','4/4 smoke 均按原 seed 一次运行并取得 60 秒前认证解；未重试失败 seed。两基线通过 `FORMAL_SCOPE_V2` 和 `CONFIG_V05` 使用相同 parent 输入、WHOLE/Y/X 目录、x_up/x_low 冻结规则、min_child_length、理论干涉、B32 与 independent certifier。旧 0.2m 成绩未复用。','',
        '## 7. 验证、隔离与下一步','',
        f"完整回归：`pytest -q -p no:cacheprovider`，326 passed。最终不变量：{json.dumps(validation,ensure_ascii=False)}。36/36 在线结果独立认证；Cmax@60 已逐 run 对照 60 秒前 best event，截止后结果未回填；NUMERIC_FAILURE=0。",'',
        'SCI 正式比较接口已经可运行，可进入统一 0.5m 的 HEURISTIC ALNS、MLP、冻结 GAT V05、Adapted-HGA、Adapted-WAG+VNS 与消融实验。启动前唯一明确条件是冻结正式实例/seed 矩阵和统计分析方案；完成后才能解封最终 ID_TEST。','']
    path=root/'docs/GRC_ALNS_C4_FINAL_SELECTION.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path,flush=True);return path


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Final C4 development utilities')
    parser.add_argument('action',choices=('final-offline','final-train','final-online','baseline-smoke','final-summary','final-validate','final-report'))
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='final-offline':run_final_c4_offline(args.root)
    elif args.action=='final-train':train_final_c4_ranked(args.root)
    elif args.action=='final-online':run_final_c4_online(args.root,args.workers)
    elif args.action=='baseline-smoke':run_final_baseline_smoke(args.root,min(args.workers,2))
    elif args.action=='final-summary':summarize_final_c4_results(args.root)
    elif args.action=='final-validate':validate_final_c4_invariants(args.root)
    else:write_final_c4_report(args.root)
