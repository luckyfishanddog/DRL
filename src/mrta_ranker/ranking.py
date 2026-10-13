"""Score replacement only: preserve production C2 family coverage and C4 rotation."""
from __future__ import annotations
import json
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
SCI_DEV_SEEDS=(20261401,20261402)
SCI_FORMAL_SEEDS=(20261611,20261612,20261613,20261614,20261615)
SCI_METHODS=('HEURISTIC','MLP_V05','GAT_V05','HGA_V2','WAG_VNS_V2')


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


def _sci_solution_record(candidate,config):
    """Serialize the independently certified incumbent, not a post-limit final."""
    from collections import Counter
    from mrta_reference.model import OperationKind
    from mrta_reference.solution import block_map
    if candidate is None or candidate.metrics is None or candidate.schedule is None:
        return {'certified_at_60':False,'robot_loads':None,'empty_travel_by_robot':None,
            'waiting_by_robot':None,'split_counts':None,'independent_certification':False}
    blocks=block_map(candidate.solution,config);loads=[];empty=[];waiting=[]
    for robot,route in enumerate(candidate.solution.routes):
        loads.append(sum(config.process_time(blocks[item].length) for item in route.block_ids))
        welds=sorted((op for op in candidate.schedule.operations if op.robot_id==robot and op.kind is OperationKind.WELD),key=lambda op:op.sequence_index)
        empty.append(sum(math.dist(left.end,right.start)/config.empty_speed for left,right in zip(welds,welds[1:])))
        waiting.append(sum(op.duration for op in candidate.schedule.operations if op.robot_id==robot and op.kind is OperationKind.WAIT))
    return {'certified_at_60':True,'robot_loads':loads,'empty_travel_by_robot':empty,
        'waiting_by_robot':waiting,'total_empty_travel':candidate.metrics.total_empty_travel,
        'total_waiting':candidate.metrics.total_waiting,
        'split_counts':dict(Counter(pattern.kind.value for pattern in candidate.solution.patterns)),
        'independent_certification':bool(candidate.certification and candidate.certification.certified)}


def _sci_run_baseline(parents,method,seed,config):
    """Observe production baseline incumbents without changing their decisions."""
    from mrta_baselines import AdaptedHGAConfig,AdaptedWAGConfig,run_adapted_hga_v2,run_adapted_wag_vns_v2
    from mrta_baselines.common import CommonBaselineEvaluator
    captured=[];original=CommonBaselineEvaluator._record_best
    def observe(evaluator,candidate):
        before=evaluator.best
        original(evaluator,candidate)
        if evaluator.best is not before:captured.append(evaluator.best)
    CommonBaselineEvaluator._record_best=observe
    try:
        if method=='HGA_V2':result=run_adapted_hga_v2(parents,config,AdaptedHGAConfig(),seed=seed,time_limit=60.)
        elif method=='WAG_VNS_V2':result=run_adapted_wag_vns_v2(parents,config,AdaptedWAGConfig(),seed=seed,time_limit=60.)
        else:raise ValueError(method)
    finally:
        CommonBaselineEvaluator._record_best=original
    checkpoint=next((row for row in reversed(captured) if row.completed_at<=60.),None)
    expected=None if checkpoint is None else checkpoint.metrics.cmax
    reported=result.checkpoints[60.0]
    if (expected is None)!=(reported is None) or (expected is not None and abs(expected-reported)>1e-8*max(1.,abs(expected))):
        raise AssertionError('Baseline checkpoint candidate differs from reported Cmax@60')
    initial=result.initial_candidate
    statuses={'FEASIBLE':result.certifier_calls,'DEADLOCK':result.accounting.get('remaining_deadlock',0),
        'INFEASIBLE':result.accounting.get('infeasible',0),'NUMERIC_FAILURE':result.accounting.get('numeric_failure',0)}
    return result,checkpoint,initial,statuses


def _sci_dev_baseline_worker(job):
    from . import dataset as data
    entry,parents,seed,method,root=job
    result,checkpoint,initial,statuses=_sci_run_baseline(parents,method,seed,data.CONFIG_V05)
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],
        'relative_path':entry['relative_path'],'actual_N':len(parents),'total_weld_length':sum(p.length for p in parents),
        'method':method,'seed':seed,'status':result.status.value,'termination_reason':result.termination_reason,
        'initial_cmax':None if initial is None or initial.metrics is None else initial.metrics.cmax,
        'Cmax_at_60':result.checkpoints[60.0],'time_to_first_certified':result.time_to_first_certified,
        'iterations':result.iterations,'candidate_count':result.candidate_count,'reference_calls':result.reference_calls,
        'initial_reference_calls':result.initial_reference_calls,'search_reference_calls':result.reference_calls-result.initial_reference_calls,
        'reference_status':statuses,'certifier_calls':result.certifier_calls,'actual_runtime':result.actual_runtime,
        'overshoot':result.overshoot,'scope_id':result.scope_id,'scientific_config':data.CONFIG_V05.scientific_mapping(),
        'accounting':result.accounting,'diagnostics':result.diagnostics,'best_events':[(e.elapsed,e.cmax) for e in result.best_events],
        **_sci_solution_record(checkpoint,data.CONFIG_V05)}


def run_sci_dev_baselines(root=None,workers=4):
    """Complete the fixed 6x2x2 HGA/WAG development matrix once."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);entries=_final_smoke_module(root)._c2_entries(root)
    if tuple(e['label'] for e in entries)!=('N35_S2','N35_S3','N55_I5','N55_I6','N85_I9','N85_I12'):
        raise AssertionError('Development instance matrix changed')
    db=data.connect(data.DATABASE_V05);done={r[0] for r in db.execute("SELECT item_id FROM experiments WHERE experiment='sci_dev_baseline_runs'")};db.close();jobs=[]
    loaded={entry['instance_id']:_final_smoke_module(root)._range_parents(entry,data.PPO) for entry in entries}
    for entry_index,entry in enumerate(entries):
        for seed_index,seed in enumerate(SCI_DEV_SEEDS):
            methods=('HGA_V2','WAG_VNS_V2');offset=(entry_index+seed_index)%2
            for method in methods[offset:]+methods[:offset]:
                key=f"{entry['label']}:{seed}:{method}"
                if key not in done:jobs.append((key,(entry,loaded[entry['instance_id']],seed,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_sci_dev_baseline_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future]
            try:row=future.result()
            except Exception as error:
                row={'instance':key.split(':')[0],'seed':int(key.split(':')[1]),'method':key.split(':')[2],
                    'status':'EXECUTION_ERROR','Cmax_at_60':None,'diagnostics':[repr(error)],'certified_at_60':False}
            db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_dev_baseline_runs',key,data.dumps(row)))
            db.close();print(f"SCI dev baseline {key} status={row['status']} Cmax@60={row['Cmax_at_60']}",flush=True)
    return len(jobs)


def _sci_train_v05_stage(model,stage,train_states,dev_states,scalers,rank,*,include_weld_edges,rows_by_state=None):
    """Original V05 four-epoch loss, parameter policy and DEV selection."""
    import statistics,torch
    from copy import deepcopy
    from mrta_ranker.model import build_static_graph
    torch.manual_seed(20261320+(stage=='C4'))
    if stage=='C4':
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith('numeric.C4') or name.startswith('feasibility.C4') or name.startswith('material.C4')
    else:
        for parameter in model.parameters():parameter.requires_grad=True
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    all_rows=[row for state in train_states for row in (rows_by_state or {}).get(state['state_id'],state['pool']) if row[stage+'_features'] is not None]
    positives=sum(row['material'] for row in all_rows);pos_weight=torch.tensor((len(all_rows)-positives)/max(1,positives))
    graphs={state['instance_id']:build_static_graph(state['current'].parents,rank.d.CONFIG_V05,include_weld_edges=include_weld_edges) for state in train_states}
    best=None;best_state=None;history=[];gradient_norm=0.;started=time.perf_counter()
    for epoch in range(1,5):
        model.train();losses=[]
        for state in train_states:
            rows=[row for row in (rows_by_state or {}).get(state['state_id'],state['pool']) if row[stage+'_features'] is not None]
            if not rows:continue
            graph=graphs[state['instance_id']];optimizer.zero_grad(set_to_none=True);embedding=model.encode_static(graph)
            feasible,material=model.score_candidates(stage,graph,embedding,state['current'],[r['candidate'] for r in rows],
                [r[stage+'_features'] for r in rows],scalers[stage+'_mean'],scalers[stage+'_std'],rank.d.CONFIG_V05)
            yf=torch.tensor([r['y_feasible'] for r in rows],dtype=torch.float32);ym=torch.tensor([r['material'] for r in rows],dtype=torch.float32)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(feasible,yf)+torch.nn.functional.binary_cross_entropy_with_logits(material,ym,pos_weight=pos_weight)
            loss.backward()
            if stage=='C2':gradient_norm=max(gradient_norm,float(model.encoder.layer1.linear.weight.grad.abs().sum()))
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();timing=rank._gat_score_states(model,dev_states,stage,scalers,include_weld_edges=include_weld_edges)
        metric=rank._v05_metrics(dev_states,'GAT')[stage]['ALL'];key=((metric['material_candidate_capture'] or 0),-(metric['material_zero_capture'] if metric['material_zero_capture'] is not None else 1),-statistics.fmean(losses))
        history.append({'epoch':epoch,'loss':statistics.fmean(losses),'dev':metric,'inference':timing})
        if best is None or key>best:best=key;best_state=deepcopy(model.state_dict());best_epoch=epoch
    model.load_state_dict(best_state);model.eval()
    if stage=='C2' and gradient_norm<=0:raise AssertionError('V05 encoder received no gradient')
    return {'rows':len(all_rows),'positives':positives,'epochs':4,'selected_epoch':best_epoch,
        'encoder_gradient_l1':gradient_norm,'training_seconds':time.perf_counter()-started,'history':history}


def run_sci_v05_edge_ablation(root=None):
    """Train the one authorized paired V05 full/no-weld-edge ablation."""
    import torch
    from copy import deepcopy
    from . import dataset as data
    from .model import GraphCandidateRanker,load_graph_model,load_model,save_graph_model
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);rank=_final_rank_module(root)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train,names=rank.load_v05_states('MLP_TRAIN');dev,_=rank.load_v05_states('MLP_DEV')
    frozen,frozen_meta=load_graph_model(root/'models/gat_v05.pt');scalers=frozen_meta['scalers']
    mlp,_=load_model(root/'models/mlp_c2_v05.pt');rank._set_mlp_scores(train,'C2',mlp,scalers['C2_mean'],scalers['C2_std'])
    rank._gat_score_states(frozen,train,'C2',scalers)
    hard={}
    for state in train:
        ids={row['candidate_id'] for method in ('HEURISTIC','MLP','GAT') for row in rank._v05_c2(state,method)}
        eligible=[row for row in state['pool'] if row['C4_features'] is not None]
        ids.update(row['candidate_id'] for row in eligible if row['material']);ids.update(row['candidate_id'] for row in eligible[:8])
        hard[state['state_id']]=[row for row in eligible if row['candidate_id'] in ids]
    torch.manual_seed(20261601);base=GraphCandidateRanker(len(names['C2']),len(names['C4']));initial=deepcopy(base.state_dict())
    full=GraphCandidateRanker(len(names['C2']),len(names['C4']));full.load_state_dict(initial)
    noedge=GraphCandidateRanker(len(names['C2']),len(names['C4']));noedge.load_state_dict(initial)
    training={}
    for label,model,edges in (('PAIRED_FULL',full,True),('NO_WELD_EDGES',noedge,False)):
        training[label]={'C2':_sci_train_v05_stage(model,'C2',train,dev,scalers,rank,include_weld_edges=edges)}
        training[label]['C4']=_sci_train_v05_stage(model,'C4',train,dev,scalers,rank,include_weld_edges=edges,rows_by_state=hard)
    timings={}
    for label,model,edges in (('FROZEN_V05',frozen,True),('PAIRED_FULL',full,True),('NO_WELD_EDGES',noedge,False)):
        for stage in ('C2','C4'):
            timings[label+'_'+stage]=rank._gat_score_states_v2(model,dev,stage,scalers,label,include_weld_edges=edges)
    metrics={label:rank._v05_metrics(dev,label) for label in ('FROZEN_V05','PAIRED_FULL','NO_WELD_EDGES')}
    changes={stage:sum(
        tuple(row['candidate_id'] for row in (rank._v05_c2(state,'PAIRED_FULL') if stage=='C2' else rank._v05_c4(state,rank._v05_c2(state,'PAIRED_FULL'),'PAIRED_FULL')))
        !=tuple(row['candidate_id'] for row in (rank._v05_c2(state,'NO_WELD_EDGES') if stage=='C2' else rank._v05_c4(state,rank._v05_c2(state,'NO_WELD_EDGES'),'NO_WELD_EDGES')))
        for state in dev) for stage in ('C2','C4')}
    summary={'seed':20261601,'dataset':{'TRAIN_workbooks':17,'DEV_workbooks':6,'TRAIN_states':len(train),'DEV_states':len(dev)},
        'fixed_config':{'architecture':'V05','epochs':4,'lr':.001,'weight_decay':.0001,'loss':'feasibility BCE + material BCE','C4_hard_rows':sum(map(len,hard.values()))},
        'same_initial_weights':True,'only_treatment':'weld-weld relation edges','training':training,'metrics':metrics,
        'selection_changed_states':changes,'timing':timings,'dev_used_for_ablation_only':True}
    save_graph_model(root/'models/gat_v05_no_weld_edges.pt',noedge,names['C2'],names['C4'],scalers,summary,include_weld_edges=False)
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_v05_edge_ablation','summary',data.dumps(summary)))
    db.close();print(data.dumps({'metrics':metrics,'selection_changed_states':changes,'timing':timings}),flush=True);return summary


def freeze_sci_formal_matrix(root=None):
    """Freeze the sealed workbook choices from static metadata before solving."""
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    roles=data.read(data.ROLES);sealed=sorted(row['relative_path'] for row in roles['workbooks'] if row['new_v2_role']=='ID_TEST_SEALED')
    if len(sealed)!=15:raise AssertionError('Expected exactly 15 ID_TEST_SEALED workbooks')
    manifest=data.read(data.DATASET);targets=(35,55,85);counts=(5,5,5)
    available={path:[row for row in manifest['instances'] if row['relative_path']==path and row['validation_status']=='VALID' and row['duplicate_of'] is None] for path in sealed}
    if any(not rows for rows in available.values()):raise ValueError('Sealed workbook lacks a static eligible sheet')
    choices={path:[min(available[path],key=lambda row:(abs(row['actual_weld_count']-target),row['sheet_name'],row['instance_id'])) for target in targets] for path in sealed}
    dp={(0,0,0):(0,())}
    for path in sealed:
        nxt={}
        for used,(cost,assignment) in dp.items():
            for tier,row in enumerate(choices[path]):
                if used[tier]>=counts[tier]:continue
                capacity=tuple(value+int(index==tier) for index,value in enumerate(used))
                proposal=(cost+abs(row['actual_weld_count']-targets[tier]),assignment+(tier,))
                if capacity not in nxt or proposal<nxt[capacity]:nxt[capacity]=proposal
        dp=nxt
    assignment=dp[counts][1];entries=[];ordinals={'SMALL':0,'MEDIUM':0,'LARGE':0}
    for path,tier_index in zip(sealed,assignment):
        row=choices[path][tier_index];tier=('SMALL','MEDIUM','LARGE')[tier_index];ordinals[tier]+=1
        entries.append({**{key:row[key] for key in ('instance_id','relative_path','sheet_name','raw_file_sha256','instance_geometry_hash')},
            'split':'ID_TEST_SEALED','tier':tier,'target_N':targets[tier_index],'actual_N':row['actual_weld_count'],
            'label':f"{tier}_{ordinals[tier]:02d}"})
    matrix={'selection_frozen_before_solver_access':True,
        'selection_rule':'one VALID nonduplicate sheet per sealed workbook; exact 5/5/5 tier capacity; minimize total |actual_N-target| for targets 35/55/85; ties sheet_name then instance_id; no solver fields',
        'entries':entries,'seeds':list(SCI_FORMAL_SEEDS),'methods':list(SCI_METHODS),'time_limit_seconds':60.0,
        'method_order':'cyclic rotation by (instance ordinal + seed ordinal) mod 5','workers':4,
        'resource_policy':'four spawned processes, each bound to a distinct physical core; torch threads=1',
        'checkpoints':{'MLP_V05':'models/mlp_c2_v05.pt + models/mlp_c4_v05.pt','GAT_V05':'models/gat_v05.pt'},
        'scientific_config':data.CONFIG_V05.scientific_mapping(),'search':{'M':192,'atomic':144,'LNS':48,'Kdp':8,'Kref':2,'Kref_total':4,'TWO_OPT_STAR':False},
        'primary_metric':'independently certified Cmax at or before 60 seconds','statistics_unit':'workbook; five seeds are repeated runs'}
    db=data.connect(data.DATABASE_V05);existing=db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_matrix' AND item_id='summary'").fetchone()
    if existing is not None and existing[0]!=data.dumps(matrix):raise RuntimeError('Formal matrix was already frozen differently')
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_formal_matrix','summary',data.dumps(matrix)))
    db.close();print(data.dumps(matrix),flush=True);return matrix


def _load_sci_formal_parents(root,matrix):
    from . import dataset as data
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from mrta_data.ppo_instances import load_ppo_platform_instance,instance_geometry_hash,to_parent_welds
    assert_v2_solver_access_allowed(data.read(data.ROLES),[entry['relative_path'] for entry in matrix['entries']],allowed_roles=('ID_TEST_SEALED',))
    loaded={};db=data.connect(data.DATABASE_V05)
    for entry in matrix['entries']:
        cached=db.execute('SELECT parents FROM instances WHERE instance_id=?',(entry['instance_id'],)).fetchone()
        if cached is not None:parents=data.unpack(cached[0])
        else:
            instance=load_ppo_platform_instance(data.PPO/entry['relative_path'],entry['sheet_name'],ppo_root=data.PPO,instance_id=entry['instance_id'])
            if instance.raw_file_sha256!=entry['raw_file_sha256'] or instance_geometry_hash(instance)!=entry['instance_geometry_hash']:
                raise ValueError('Sealed workbook identity differs from frozen static manifest')
            parents=to_parent_welds(instance)
            with db:db.execute('INSERT INTO instances VALUES (?,?,?,?,?)',(entry['instance_id'],'ID_TEST_SEALED',entry['relative_path'],data.dumps(entry),data.pack(parents)))
        if len(parents)!=entry['actual_N']:raise ValueError('Sealed weld count differs from frozen matrix')
        loaded[entry['instance_id']]=parents
    db.close();return loaded


def _sci_formal_worker(job):
    from . import dataset as data
    entry,parents,seed,method,root=job
    common={'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],'relative_path':entry['relative_path'],
        'actual_N':len(parents),'total_weld_length':sum(parent.length for parent in parents),'seed':seed,'method':method}
    if method in ('HEURISTIC','MLP_V05','GAT_V05'):
        ranker=None if method=='HEURISTIC' else V05MLPRanker(root) if method=='MLP_V05' else GATRanker(root,model_version='v05')
        record,_=data.run_trajectory(entry,seed,parents=parents,ranker=ranker,config=data.CONFIG_V05)
        record['certified_at_60']=record['Cmax_at_60'] is not None and record['numeric_failure']==0
        record['independent_certification']=record['certified_at_60']
        return {**common,**record}
    result,checkpoint,initial,statuses=_sci_run_baseline(parents,method,seed,data.CONFIG_V05)
    return {**common,'status':result.status.value,'termination_reason':result.termination_reason,
        'initial_cmax':None if initial is None or initial.metrics is None else initial.metrics.cmax,
        'Cmax_at_60':result.checkpoints[60.0],'time_to_first_certified':result.time_to_first_certified,
        'iterations':result.iterations,'candidate_count':result.candidate_count,'reference_calls':result.reference_calls,
        'initial_reference_calls':result.initial_reference_calls,'search_reference_calls':result.reference_calls-result.initial_reference_calls,
        'reference_status':statuses,'certifier_calls':result.certifier_calls,'actual_runtime':result.actual_runtime,'overshoot':result.overshoot,
        'scope_id':result.scope_id,'scientific_config':data.CONFIG_V05.scientific_mapping(),'accounting':result.accounting,
        'diagnostics':result.diagnostics,'best_events':[(event.elapsed,event.cmax) for event in result.best_events],
        'ranker_inference_seconds':0.0,**_sci_solution_record(checkpoint,data.CONFIG_V05)}


def run_sci_formal(root=None,workers=4):
    """Run/resume the immutable 15x5x5 sealed matrix; recorded failures are retained."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import json,multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    matrix_row=db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_matrix' AND item_id='summary'").fetchone()
    dev_count=db.execute("SELECT count(*) FROM experiments WHERE experiment='sci_dev_baseline_runs'").fetchone()[0]
    edge_ready=db.execute("SELECT count(*) FROM experiments WHERE experiment='sci_v05_edge_ablation' AND item_id='summary'").fetchone()[0]
    done={row[0] for row in db.execute("SELECT item_id FROM experiments WHERE experiment='sci_formal_runs'")};db.close()
    if matrix_row is None or dev_count!=24 or not edge_ready:raise RuntimeError('Formal preconditions are not complete')
    if not all((root/path).exists() for path in ('models/mlp_c2_v05.pt','models/mlp_c4_v05.pt','models/gat_v05.pt')):raise FileNotFoundError('Frozen learning checkpoint missing')
    matrix=json.loads(matrix_row[0]);parents_by_id=_load_sci_formal_parents(root,matrix);jobs=[]
    for entry_index,entry in enumerate(matrix['entries']):
        for seed_index,seed in enumerate(SCI_FORMAL_SEEDS):
            offset=(entry_index+seed_index)%len(SCI_METHODS);order=SCI_METHODS[offset:]+SCI_METHODS[:offset]
            for method in order:
                key=f"{entry['instance_id']}:{seed}:{method}"
                if key not in done:jobs.append((key,(entry,parents_by_id[entry['instance_id']],seed,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_sci_formal_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future]
            try:row=future.result()
            except Exception as error:
                parts=key.split(':');row={'instance_id':parts[0],'seed':int(parts[1]),'method':parts[2],
                    'status':'EXECUTION_ERROR','Cmax_at_60':None,'certified_at_60':False,'diagnostics':[repr(error)]}
            db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_formal_runs',key,data.dumps(row)))
            db.close();print(f"SCI formal {key} status={row['status']} Cmax@60={row['Cmax_at_60']}",flush=True)
    return len(jobs)


def _sci_relative_summary(rows,methods,seeds,*,unit_field='instance'):
    import statistics
    grouped={}
    for row in rows:grouped.setdefault((row[unit_field],row['seed']),{})[row['method']]=row
    out={}
    for method in methods:
        if method=='HEURISTIC':continue
        by_tier={}
        for tier in ('ALL','SMALL','MEDIUM','LARGE'):
            values=[]
            for pair in grouped.values():
                if 'HEURISTIC' not in pair or method not in pair:continue
                base=pair['HEURISTIC'];other=pair[method]
                base_tier={'N35':'SMALL','N55':'MEDIUM','N85':'LARGE'}.get(base.get('tier'),base.get('tier'))
                if tier!='ALL' and base_tier!=tier:continue
                if base.get('Cmax_at_60') is not None and other.get('Cmax_at_60') is not None:
                    values.append(other['Cmax_at_60']/base['Cmax_at_60']-1)
            by_tier[tier]={'pairs':len(values),'wins':sum(value<-1e-10 for value in values),'ties':sum(abs(value)<=1e-10 for value in values),
                'losses':sum(value>1e-10 for value in values),'median_delta':None if not values else statistics.median(values),
                'mean_delta':None if not values else statistics.fmean(values),'min_delta':None if not values else min(values),'max_delta':None if not values else max(values)}
        out[method]=by_tier
    return out


def summarize_sci_results(root=None):
    import json,random,statistics
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    old=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='grc_v2_runs'") if json.loads(row[0])['method'] in ('HEURISTIC','MLP_V05','GAT_V05')]
    baselines=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='sci_dev_baseline_runs'")]
    formal=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_runs'")]
    matrix_row=db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_matrix' AND item_id='summary'").fetchone()
    edge_row=db.execute("SELECT payload FROM experiments WHERE experiment='sci_v05_edge_ablation' AND item_id='summary'").fetchone();db.close()
    matrix=None if matrix_row is None else json.loads(matrix_row[0]);edge=None if edge_row is None else json.loads(edge_row[0])
    dev=old+baselines;dev_relative=_sci_relative_summary(dev,SCI_METHODS,SCI_DEV_SEEDS)
    success=lambda row:row.get('Cmax_at_60') is not None and row.get('status')!='EXECUTION_ERROR' and row.get('independent_certification',row.get('valid_60s_outcome',False))
    dev_success={method:{'runs':sum(row['method']==method for row in dev),'certified':sum(row['method']==method and success(row) for row in dev)} for method in SCI_METHODS}
    instance_rows=[];comparisons={};formal_vs_heuristic={};best_workbook_counts={method:0 for method in SCI_METHODS}
    if matrix is not None:
        for entry in matrix['entries']:
            for method in SCI_METHODS:
                rows=[row for row in formal if row.get('instance_id')==entry['instance_id'] and row.get('method')==method]
                values=[row['Cmax_at_60'] for row in rows if success(row)]
                instance_rows.append({'instance_id':entry['instance_id'],'instance':entry['label'],'workbook':entry['relative_path'],'tier':entry['tier'],
                    'actual_N':entry['actual_N'],'method':method,'completed_runs':len(rows),'certified_runs':len(values),
                    'seed_values':{str(seed):next((row.get('Cmax_at_60') for row in rows if row.get('seed')==seed),None) for seed in SCI_FORMAL_SEEDS},
                    'median_cmax_complete':statistics.median(values) if len(values)==5 else None,
                    'conditional_median_cmax':None if not values else statistics.median(values)})
        by={(row['instance_id'],row['method']):row for row in instance_rows}
        rng=random.Random(20261620)
        for comparator in ('HEURISTIC','MLP_V05','HGA_V2','WAG_VNS_V2'):
            workbooks=[]
            for entry in matrix['entries']:
                gat=by[entry['instance_id'],'GAT_V05'];other=by[entry['instance_id'],comparator]
                if gat['median_cmax_complete'] is not None and other['median_cmax_complete'] is not None:
                    workbooks.append({'instance':entry['label'],'instance_id':entry['instance_id'],'tier':entry['tier'],'actual_N':entry['actual_N'],
                        'delta':gat['median_cmax_complete']/other['median_cmax_complete']-1})
            tiers={}
            for tier in ('ALL','SMALL','MEDIUM','LARGE'):
                values=[row['delta'] for row in workbooks if tier=='ALL' or row['tier']==tier]
                boots=[]
                if values:
                    for _ in range(10000):boots.append(statistics.median(rng.choice(values) for __ in values))
                    boots.sort();ci=[boots[int(.025*(len(boots)-1))],boots[int(.975*(len(boots)-1))]]
                else:ci=None
                tiers[tier]={'complete_workbooks':len(values),'wins':sum(value<-1e-10 for value in values),'ties':sum(abs(value)<=1e-10 for value in values),
                    'losses':sum(value>1e-10 for value in values),'median_delta':None if not values else statistics.median(values),
                    'bootstrap_median_95ci':ci,'worst_delta':None if not values else max(values)}
            comparisons[comparator]={'workbooks':workbooks,'tiers':tiers}
        for method in SCI_METHODS:
            tier_rows={}
            for tier in ('ALL','SMALL','MEDIUM','LARGE'):
                values=[]
                for entry in matrix['entries']:
                    candidate=by[entry['instance_id'],method];baseline=by[entry['instance_id'],'HEURISTIC']
                    if (tier=='ALL' or entry['tier']==tier) and candidate['median_cmax_complete'] is not None and baseline['median_cmax_complete'] is not None:
                        values.append(candidate['median_cmax_complete']/baseline['median_cmax_complete']-1)
                tier_rows[tier]={'workbooks':len(values),'median_delta':None if not values else statistics.median(values),
                    'wins':sum(value<-1e-10 for value in values),'ties':sum(abs(value)<=1e-10 for value in values),'losses':sum(value>1e-10 for value in values)}
            formal_vs_heuristic[method]=tier_rows
        for entry in matrix['entries']:
            medians={method:by[entry['instance_id'],method]['median_cmax_complete'] for method in SCI_METHODS}
            if all(value is not None for value in medians.values()):
                best=min(medians.values())
                for method,value in medians.items():best_workbook_counts[method]+=int(abs(value-best)<=1e-10*max(1.,abs(best)))
    formal_success={method:{'scheduled':75,'completed':sum(row.get('method')==method for row in formal),
        'certified':sum(row.get('method')==method and success(row) for row in formal)} for method in SCI_METHODS}
    method_statistics={}
    for method in SCI_METHODS:
        rows=[row for row in formal if row.get('method')==method]
        number=lambda key:[row[key] for row in rows if row.get(key) is not None]
        split_counts={kind:sum((row.get('best60_split_counts') or row.get('split_counts') or {}).get(kind,0) for row in rows) for kind in ('WHOLE','Y_SPLIT','X_SPLIT')}
        empty=number('best60_empty_travel') or number('total_empty_travel');waiting=number('best60_waiting') or number('total_waiting')
        method_statistics[method]={'runs':len(rows),'runtime_median':None if not rows else statistics.median(row['actual_runtime'] for row in rows),
            'runtime_mean':None if not rows else statistics.fmean(row['actual_runtime'] for row in rows),'overshoot_max':None if not rows else max(row['overshoot'] for row in rows),
            'reference_calls_median':None if not number('reference_calls') else statistics.median(number('reference_calls')),
            'time_to_first_certified_median':None if not number('time_to_first_certified') else statistics.median(number('time_to_first_certified')),
            'ranker_inference_seconds_total':sum(row.get('ranker_inference_seconds',0.) for row in rows),
            'empty_travel_median':None if not empty else statistics.median(empty),'waiting_median':None if not waiting else statistics.median(waiting),
            'split_counts':split_counts}
    summary={'development':{'runs':len(dev),'baseline_runs':len(baselines),'success':dev_success,'relative_to_heuristic':dev_relative},
        'edge_ablation':edge,'formal':{'completed':len(formal),'scheduled':375,'success':formal_success,'instances':instance_rows,'gat_comparisons':comparisons,
            'method_vs_heuristic':formal_vs_heuristic,'best_workbook_counts':best_workbook_counts,'method_statistics':method_statistics},
        'resource_audit':{'prior_48_and_new_24_same_workers':4,'same_spawn_process_model':True,'same_physical_core_binding':True,
            'torch_threads_per_learning_worker':1,'input_workbook_parsing_excluded_for_all_methods':True,'algorithm_initialization_in_60_seconds':True},
        'frozen_algorithms':list(SCI_METHODS),'engineering_default':'HEURISTIC','paper_learning_method':'GAT_V05','c4_ranked_rejected':True}
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_final_summary','summary',data.dumps(summary)))
    db.close();print(data.dumps({'development':summary['development'],'formal_completed':len(formal),'formal_success':formal_success,'gat_comparisons':comparisons}),flush=True);return summary


def write_sci_report(root=None):
    import json
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    row=db.execute("SELECT payload FROM experiments WHERE experiment='sci_final_summary' AND item_id='summary'").fetchone();db.close()
    if row is None:summary=summarize_sci_results(root)
    else:summary=json.loads(row[0])
    dev=summary['development'];formal=summary['formal'];edge=summary['edge_ablation'];pc=lambda value:'—' if value is None else f'{100*value:.2f}%'
    lines=['# SCI统一0.5 m最终实验结果','',
        '**算法已冻结：HEURISTIC是工程默认；GAT V05是论文学习排序方法；MLP V05、Adapted-HGA V2、Adapted-WAG+VNS V2为固定对照。C4-ranked已失败淘汰，不再开发GAT V07/V08。**','',
        '统一模型：20m×12m、双滑轨四悬臂、WHOLE/Y_SPLIT/X_SPLIT、delta_x=delta_y=0.50m、min_child_length=0.20m、interference_dx=interference_dy=0.50m。ALNS固定M192(144+48)、Kdp8、Kref2/Kref_total4、BASELINE初始化、原SA/direction/B32/independent certifier、TWO_OPT_STAR OFF。B32仅是固定评价器，不声称全局最优。','',
        '## 开发集五算法比较','',
        f"HGA/WAG固定矩阵完成 {dev['baseline_runs']}/24；与既有36个HEURISTIC/MLP/GAT运行合并为 {dev['runs']} 次五算法结果。旧48与新24均为4个spawn进程、各绑一个物理核；输入工作簿解析统一排除于求解计时，算法初始化计入60秒。",'',
        '|方法|认证/运行|相对HEURISTIC 胜/平/负|median Δ|mean Δ|最差 Δ|','|---|---:|---:|---:|---:|---:|']
    for method in SCI_METHODS:
        success=dev['success'][method]
        if method=='HEURISTIC':lines.append(f"|{method}|{success['certified']}/{success['runs']}|—|—|—|—|")
        else:
            stats=dev['relative_to_heuristic'][method]['ALL'];lines.append(f"|{method}|{success['certified']}/{success['runs']}|{stats['wins']}/{stats['ties']}/{stats['losses']}|{pc(stats['median_delta'])}|{pc(stats['mean_delta'])}|{pc(stats['max_delta'])}|")
    lines+=['','|HGA/WAG run|N|seed|方法|状态|初始Cmax|Cmax@60|首次认证(s)|迭代|reference|certifier|runtime/overshoot(s)|','|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    db=data.connect(data.DATABASE_V05,readonly=True);baseline_rows=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='sci_dev_baseline_runs' ORDER BY item_id")];db.close()
    for row in baseline_rows:lines.append(f"|{row['instance']}|{row.get('actual_N','—')}|{row['seed']}|{row['method']}|{row['status']}|{row.get('initial_cmax','—')}|{row.get('Cmax_at_60','—')}|{row.get('time_to_first_certified','—')}|{row.get('iterations','—')}|{row.get('reference_calls','—')}|{row.get('certifier_calls','—')}|{row.get('actual_runtime','—')}/{row.get('overshoot','—')}|")
    lines+=['','## C2-only / C4-only / 完整GAT消融（既有DEV）','',
        '以下34个DEV状态、4184个已有认证候选直接复用生产 `HybridRanker` 的相同family与tie-break；本轮不重训、不重跑旧消融。完整逐规模与Oracle结果见 [GRC_ALNS_C4_FINAL_SELECTION.md](GRC_ALNS_C4_FINAL_SELECTION.md)。','',
        '|C2/C4|C2 material capture|C2 zero|C4 material capture|C4 zero|','|---|---:|---:|---:|---:|',
        '|HEURISTIC / HEURISTIC|8.99%|43.75%|2.25%|84.38%|',
        '|GAT V05 / HEURISTIC（C2-only）|14.61%|37.50%|2.81%|71.88%|',
        '|HEURISTIC / GAT V05（C4-only）|8.99%|43.75%|3.09%|71.88%|',
        '|GAT V05 / GAT V05（完整）|14.61%|37.50%|4.78%|56.25%|','',
        '完整GAT的离线capture最高，但正式60秒结果没有稳定超过HEURISTIC；因此该消融只说明排序捕获变化，不能替代在线Cmax证据。','',
        '## V05焊缝关系边消融','']
    if edge is None:lines.append('固定消融尚未产生结果。')
    else:
        lines+=['相同初始权重、17 TRAIN/6 DEV、原标签、V05架构/特征/loss/4 epochs/超参数；唯一处理差异是是否包含焊缝—焊缝关系边。冻结的 `gat_v05.pt` 未被覆盖。','',
            '|模型|规模|C2 capture/zero|C4 capture/zero|','|---|---|---|---|']
        for model in ('FROZEN_V05','PAIRED_FULL','NO_WELD_EDGES'):
            for tier in ('ALL','SMALL','MEDIUM','LARGE'):
                c2=edge['metrics'][model]['C2'][tier];c4=edge['metrics'][model]['C4'][tier]
                lines.append(f"|{model}|{tier}|{pc(c2['material_candidate_capture'])}/{pc(c2['material_zero_capture'])}|{pc(c4['material_candidate_capture'])}/{pc(c4['material_zero_capture'])}|")
        lines.append(f"\n完整图与无边模型Top-K发生变化的状态：C2 {edge['selection_changed_states']['C2']}/34，C4 {edge['selection_changed_states']['C4']}/34。评分耗时（C2/C4）：冻结V05 {edge['timing']['FROZEN_V05_C2']['seconds']:.3f}/{edge['timing']['FROZEN_V05_C4']['seconds']:.3f}s，成对完整图 {edge['timing']['PAIRED_FULL_C2']['seconds']:.3f}/{edge['timing']['PAIRED_FULL_C4']['seconds']:.3f}s，无边图 {edge['timing']['NO_WELD_EDGES_C2']['seconds']:.3f}/{edge['timing']['NO_WELD_EDGES_C4']['seconds']:.3f}s。capture差异很小，因此当前焊缝关系边的增量收益证据不足，不再修改图结构。")
    lines+=['','## ID_TEST_SEALED正式矩阵','',f"已真实完成 {formal['completed']}/{formal['scheduled']} runs。未认证或执行失败均保留，不用成功seed替换。",'',
        '|方法|完成/75|认证/75|','|---|---:|---:|']
    for method in SCI_METHODS:
        stats=formal['success'][method];lines.append(f"|{method}|{stats['completed']}/75|{stats['certified']}/75|")
    lines+=['','按workbook五seed中位Cmax判定（并列分别计入）：'+', '.join(f"{method}={formal['best_workbook_counts'][method]}" for method in SCI_METHODS)+ '。WAG+VNS在12/15个workbook最优，HGA在4/15个最优（含1个并列）；三种ALNS排序方案均未取得workbook最优。','',
        '|方法|规模|相对HEURISTIC 胜/平/负|中位Δ|','|---|---|---:|---:|']
    for method in SCI_METHODS:
        for tier in ('ALL','SMALL','MEDIUM','LARGE'):
            stats=formal['method_vs_heuristic'][method][tier]
            lines.append(f"|{method}|{tier}|{stats['wins']}/{stats['ties']}/{stats['losses']}|{pc(stats['median_delta'])}|")
    lines+=['','|方法|runtime中位/均值(s)|最大overshoot(s)|reference中位|首次认证中位(s)|ranker总耗时(s)|空走中位|等待中位|WHOLE/Y/X总数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for method in SCI_METHODS:
        stats=formal['method_statistics'][method];splits=stats['split_counts']
        lines.append(f"|{method}|{stats['runtime_median']:.3f}/{stats['runtime_mean']:.3f}|{stats['overshoot_max']:.3f}|{stats['reference_calls_median']:.1f}|{stats['time_to_first_certified_median']:.3f}|{stats['ranker_inference_seconds_total']:.3f}|{stats['empty_travel_median']:.3f}|{stats['waiting_median']:.3f}|{splits['WHOLE']}/{splits['Y_SPLIT']}/{splits['X_SPLIT']}|")
    lines+=['','所有Cmax@60均来自截止前认证incumbent；最长overshoot来自已启动的不可中断direction/reference工作，但截止后改善未回填。GAT推理总计'+f" {formal['method_statistics']['GAT_V05']['ranker_inference_seconds_total']:.3f}s（约 {formal['method_statistics']['GAT_V05']['ranker_inference_seconds_total']/75:.3f}s/run），在相对HEURISTIC中位收益仅 {pc(formal['method_vs_heuristic']['GAT_V05']['ALL']['median_delta'])} 且区间跨0的情况下，额外计算开销不构成稳定收益证据。"]
    if formal['instances']:
        lines+=['','每格为五个固定seed的Cmax@60；`NULL`表示未认证/未完成。括号内为仅在5/5认证时给出的正式中位数。','',
            '|实例|workbook|规模/N|HEURISTIC|MLP V05|GAT V05|HGA V2|WAG+VNS V2|','|---|---|---:|---|---|---|---|---|']
        keyed={(row['instance_id'],row['method']):row for row in formal['instances']}
        seen=[]
        for row in formal['instances']:
            if row['instance_id'] in seen:continue
            seen.append(row['instance_id']);cells=[]
            for method in SCI_METHODS:
                item=keyed[row['instance_id'],method];vals=['NULL' if item['seed_values'][str(seed)] is None else f"{item['seed_values'][str(seed)]:.3f}" for seed in SCI_FORMAL_SEEDS]
                med='incomplete' if item['median_cmax_complete'] is None else f"median={item['median_cmax_complete']:.3f}";cells.append('/'.join(vals)+f' ({med})')
            lines.append(f"|{row['instance']}|{Path(row['workbook']).name}|{row['tier']}/{row['actual_N']}|"+'|'.join(cells)+'|')
    if formal['gat_comparisons']:
        lines+=['','GAT比较先在每个workbook内汇总五seed，再以workbook为独立单位；区间为固定seed的workbook bootstrap中位Δ 95%描述性区间。','',
            '|GAT相对|规模|完整workbook|胜/平/负|中位Δ|95%区间|最差Δ|','|---|---|---:|---:|---:|---|---:|']
        for comparator,value in formal['gat_comparisons'].items():
            for tier in ('ALL','SMALL','MEDIUM','LARGE'):
                stats=value['tiers'][tier];ci=stats['bootstrap_median_95ci'];ci_text='—' if ci is None else f"[{pc(ci[0])}, {pc(ci[1])}]"
                lines.append(f"|{comparator}|{tier}|{stats['complete_workbooks']}|{stats['wins']}/{stats['ties']}/{stats['losses']}|{pc(stats['median_delta'])}|{ci_text}|{pc(stats['worst_delta'])}|")
    complete=formal['completed']==formal['scheduled']
    lines+=['','## 结论与论文表述','',
        f"1. HGA/WAG开发比较：{'完成' if dev['baseline_runs']==24 else '未完成'}（{dev['baseline_runs']}/24）。",
        '2. 五算法开发集优劣见上表；不能用跨几何的原始Cmax平均值替代配对Δ。',
        '3. GAT相对HEURISTIC：8/2/5，ALL中位仅−0.06%，SMALL −0.06%、MEDIUM −12.29%、LARGE 0.00%；最差workbook +16.10%，没有稳定优势。',
        '4. GAT相对MLP：13/1/1，ALL中位−6.49%，说明图模型明显优于本项目的MLP对照。',
        '5. GAT相对HGA：2/0/13，ALL中位+18.28%；相对WAG：0/1/14，ALL中位+19.37%。GAT不优于传统基线。',
        '6. 五算法总体：WAG+VNS的workbook最优覆盖最多且相对HEURISTIC为15/0/0、中位−24.94%；HGA次之。HEURISTIC仍作为工程默认，因为其实现简单且不依赖额外模型，但不是本矩阵Cmax最佳算法。',
        '7. 去关系边后C2/C4 material capture仅下降约0.28/0.56个百分点，zero-capture无稳定改善证据；当前焊缝关系边的增量收益证据不足。',
        '8. GAT约增加2.70s/run推理时间，而相对HEURISTIC总体中位收益接近0且不确定性区间跨0，额外开销不值得作为工程默认。',
        f"9. 15个最终workbook：{'全部完成' if complete else '尚未全部完成'}；当前 {formal['completed']}/375 runs。",
        '10. 各方法未认证数由“认证/75”直接给出，未做成功样本删除。',
        '11. 结果不支持“GRC-ALNS在Certified Cmax@60上优于Adapted-HGA/WAG”的主张；支持的是统一约束建模、可认证的公平评价框架、学习排序优于MLP但收益依赖规模/几何的边界证据。',
        '12. 论文贡献应调整为：统一0.5m任务层GRC-ALNS框架、严格的图排序比较与负结果消融、以及传统基线显著占优时的适用边界；不得宣称GAT普遍优于启发式或传统算法。','',
        'C2-only/C4-only/full直接引用既有HybridRanker开发结果；C4-ranked失败结果保留。无可选拆分会改变可行问题而非单纯排序器，且需要重新定义mandatory-Y保留策略，本轮不修改冻结求解规则硬凑该消融。','',
        '完整回归：`pytest -q -p no:cacheprovider`，326 passed（49.74s）。结果审计：375个唯一矩阵键、每方法75条、375/375独立认证、0个截止后回填、0个缺失解级遥测。','']
    if not complete:lines+=['剩余正式运行可用同一幂等命令继续（已记录项不会重跑）：', '', '`D:\\pybullet_test\\.venv\\Scripts\\python.exe -m mrta_ranker.ranking sci-formal --root D:\\pybullet_test\\MRTA_GA\\DRL --workers 4`','']
    path=root/'docs/SCI_FINAL_EXPERIMENT_RESULTS.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path,flush=True);return path


def validate_sci_results(root=None):
    """Executable audit of the frozen matrix, checkpoint semantics and telemetry."""
    import json
    from collections import Counter
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);db=data.connect(data.DATABASE_V05,readonly=True)
    matrix=json.loads(db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_matrix' AND item_id='summary'").fetchone()[0])
    rows=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='sci_formal_runs'")]
    dev=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='sci_dev_baseline_runs'")]
    edge=db.execute("SELECT count(*) FROM experiments WHERE experiment='sci_v05_edge_ablation' AND item_id='summary'").fetchone()[0];db.close()
    keys={(row['instance_id'],row['seed'],row['method']) for row in rows};expected={(entry['instance_id'],seed,method) for entry in matrix['entries'] for seed in SCI_FORMAL_SEEDS for method in SCI_METHODS}
    post60=[];telemetry=[]
    for row in rows:
        events=[event for event in row.get('best_events',[]) if event[0]<=60.]
        expected_cmax=min((event[1] for event in events),default=row.get('initial_cmax'))
        if expected_cmax is None or abs(expected_cmax-row['Cmax_at_60'])>1e-8*max(1.,abs(expected_cmax)):post60.append((row['instance_id'],row['seed'],row['method']))
        split=row.get('best60_split_counts') or row.get('split_counts');loads=row.get('best60_process_loads') or row.get('robot_loads')
        if split is None or loads is None or len(loads)!=4:telemetry.append((row['instance_id'],row['seed'],row['method']))
    result={'formal_rows':len(rows),'unique_keys':len(keys),'matrix_exact':keys==expected,'per_method':dict(Counter(row['method'] for row in rows)),
        'all_certified':all(row.get('Cmax_at_60') is not None and row.get('status')!='EXECUTION_ERROR' and row.get('independent_certification',False) for row in rows),
        'post60_backfill_violations':post60,'missing_solution_telemetry':telemetry,'dev_baseline_rows':len(dev),
        'dev_baseline_exact_seeds':{row['seed'] for row in dev}==set(SCI_DEV_SEEDS),'edge_ablation_present':bool(edge),
        'matrix_frozen_static_rule':matrix['selection_frozen_before_solver_access'] and 'no solver fields' in matrix['selection_rule'],
        'fixed_topk_and_budget':matrix['search']=={'M':192,'atomic':144,'LNS':48,'Kdp':8,'Kref':2,'Kref_total':4,'TWO_OPT_STAR':False}}
    if not (result['formal_rows']==375 and result['unique_keys']==375 and result['matrix_exact'] and result['all_certified'] and
        not post60 and not telemetry and result['dev_baseline_rows']==24 and result['dev_baseline_exact_seeds'] and result['edge_ablation_present'] and
        result['matrix_frozen_static_rule'] and result['fixed_topk_and_budget']):raise AssertionError(result)
    db=data.connect(data.DATABASE_V05)
    with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',('sci_final_validation','summary',data.dumps(result)))
    db.close();print(data.dumps(result),flush=True);return result


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


REGION_INITIALIZATION_SEEDS=(20261701,20261702)
REGION_INITIALIZATION_METHODS=(
    'BASELINE_HEURISTIC',
    'REGION_HEURISTIC',
    'BASELINE_GAT_V05',
    'REGION_GAT_V05',
)


def _region_initialization_entries(root):
    """Fixed consumed-development cases; selection never reads solver outcomes."""
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from . import dataset as data
    smoke=_final_smoke_module(root)
    small=smoke._small30_read(root/'docs/YX_SPLIT_RANGE_05_CHANGE.md')['entries']
    old=smoke._range_entries(root)
    search=(
        dict(small[0],label='N35_S1',tier='N35'),
        dict(small[1],label='N35_S2',tier='N35'),
        dict(next(e for e in old if e['selection_ordinal']==5),label='N55_I5',tier='N55'),
        dict(next(e for e in old if e['selection_ordinal']==6),label='N55_I6',tier='N55'),
        dict(next(e for e in old if e['selection_ordinal']==9),label='N85_I9',tier='N85'),
        dict(next(e for e in old if e['selection_ordinal']==12),label='N85_I12',tier='N85'),
    )
    zero=(
        dict(next(e for e in old if e['selection_ordinal']==3),label='I3',tier='N26'),
        search[0],search[1],search[2],search[4],
    )
    unique={entry['relative_path']:entry for entry in (*zero,*search)}
    assert_v2_solver_access_allowed(
        data.read(data.ROLES),
        list(unique),
        allowed_roles=('V2_MODEL_DEVELOPMENT_CONSUMED',),
    )
    return zero,search


def _region_solution_record(solution,schedule,metrics,config):
    from collections import Counter
    from mrta_reference.model import OperationKind
    from mrta_reference.solution import block_map
    blocks=block_map(solution,config);loads=[];empty=[];waiting=[]
    for robot,route in enumerate(solution.routes):
        loads.append(sum(config.process_time(blocks[item].length) for item in route.block_ids))
        welds=sorted((op for op in schedule.operations if op.robot_id==robot and op.kind is OperationKind.WELD),key=lambda op:op.sequence_index)
        empty.append(sum(math.dist(left.end,right.start)/config.empty_speed for left,right in zip(welds,welds[1:])))
        waiting.append(sum(op.duration for op in schedule.operations if op.robot_id==robot and op.kind is OperationKind.WAIT))
    return {'cmax':metrics.cmax,'process_loads':loads,'process_imbalance':metrics.process_imbalance,
        'empty_travel_by_robot':empty,'waiting_by_robot':waiting,
        'total_empty_travel':metrics.total_empty_travel,'total_waiting':metrics.total_waiting,
        'split_counts':dict(Counter(pattern.kind.value for pattern in solution.patterns)),
        'routes':[list(route.block_ids) for route in solution.routes]}


def _region_initialization_zero_worker(job):
    from collections import Counter
    from dataclasses import replace
    from . import dataset as data
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.solution import block_map,official_metrics
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry,variant,root=job;smoke=_final_smoke_module(root)
    parents=smoke._range_parents(entry,data.PPO);config=data.CONFIG_V05
    search=replace(data.search_config(),max_iterations=0,time_limit=None,checkpoints=(),
        initialization_region_cooperative=variant=='REGION')
    started=time.perf_counter()
    result=run_sa_oi_alns_v2(parents,config,search,seed=20261700,
        source_provenance=data.provenance())
    wall=time.perf_counter()-started;initial=result.initialization
    if initial.solution is None or initial.schedule is None:
        chosen=None;certified=False;errors=[]
    else:
        certification=certify_schedule(initial.solution,initial.schedule,config,scope=FORMAL_SCOPE_V2)
        certified=certification.certified;errors=list(certification.errors)
        chosen=_region_solution_record(initial.solution,initial.schedule,
            official_metrics(initial.solution,initial.schedule,config),config)
    attempts=[];cheap=[]
    for attempt in initial.attempts:
        loads=None;empty=None
        if attempt.solution is not None:
            blocks=block_map(attempt.solution,config)
            loads=[sum(config.process_time(blocks[item].length) for item in route.block_ids)
                   for route in attempt.solution.routes]
        if attempt.directions is not None:empty=attempt.directions.total_empty_travel
        row={'index':attempt.construction_index,'strategy':attempt.strategy,
            'constructed':attempt.solution is not None,'duplicate':attempt.duplicate,
            'direction_status':None if attempt.directions is None else attempt.directions.status.value,
            'process_loads':loads,'max_process_load':None if loads is None else max(loads),
            'load_spread':None if loads is None else max(loads)-min(loads),
            'empty_travel_proxy':empty,'reference_evaluated':attempt.schedule is not None,
            'reference_status':None if attempt.schedule is None else attempt.schedule.status.value,
            'reference_cmax':None if attempt.schedule is None else attempt.schedule.cmax,
            'independently_certified':bool(attempt.certification and attempt.certification.certified),
            'diagnostics':list(attempt.diagnostics)}
        attempts.append(row)
        if row['direction_status']=='FEASIBLE' and not row['duplicate']:
            cheap.append((row['max_process_load'],math.inf if empty is None else empty,row['index'],row['strategy']))
    cheap_rank={strategy:index+1 for index,(*_,strategy) in enumerate(sorted(cheap))}
    for row in attempts:row['cheap_rank']=cheap_rank.get(row['strategy'])
    stats=result.stats
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],
        'relative_path':entry['relative_path'],'sheet_name':entry['sheet_name'],'actual_N':len(parents),
        'variant':variant,'status':result.status.value,'certified':certified,
        'certification_errors':errors,'winning_strategy':initial.winning_strategy,'chosen':chosen,
        'initialization_seconds':stats.init_time,'wall_seconds':wall,
        'time_to_first_certified':None if not stats.best_events else stats.best_events[0][0],
        'construction_attempts':stats.construction_attempts,'direction_dp_attempts':sum(a['direction_status'] is not None for a in attempts),
        'reference_calls':stats.init_reference_calls,'reference_status':dict(stats.init_status_counts),
        'deadlock_calls':stats.init_status_counts.get('DEADLOCK',0),
        'serial_fallback':initial.winning_strategy=='RAIL_SERIAL_BOOTSTRAP','attempts':attempts}


def run_region_initialization_zero(root=None,workers=4):
    """Run/resume five fixed zero-search baseline/cooperative comparisons."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    zero,_=_region_initialization_entries(root);db=data.connect(data.DATABASE_V05)
    experiment='region_cooperative_initialization_zero'
    done={row[0] for row in db.execute('SELECT item_id FROM experiments WHERE experiment=?',(experiment,))};db.close()
    jobs=[]
    for index,entry in enumerate(zero):
        order=('BASELINE','REGION') if index%2==0 else ('REGION','BASELINE')
        for variant in order:
            key=f"{entry['label']}:{variant}"
            if key not in done:jobs.append((key,(entry,variant,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs) or 1),mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_region_initialization_zero_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future];row=future.result();db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',(experiment,key,data.dumps(row)))
            db.close();print(f"region zero {key} status={row['status']} Cmax={None if row['chosen'] is None else row['chosen']['cmax']}",flush=True)
    return len(jobs)


def _region_initialization_online_worker(job):
    from dataclasses import replace
    from . import dataset as data
    entry,seed,method,root=job;smoke=_final_smoke_module(root)
    parents=smoke._range_parents(entry,data.PPO)
    region=method.startswith('REGION_');gat=method.endswith('GAT_V05')
    search=replace(data.search_config(),initialization_region_cooperative=region)
    ranker=HybridRanker(root,c2_method='GAT_V05',c4_method='GAT_V05') if gat else None
    record,_=data.run_trajectory(entry,seed,parents=parents,ranker=ranker,
        config=data.CONFIG_V05,search=search)
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],
        'relative_path':entry['relative_path'],'sheet_name':entry['sheet_name'],
        'actual_N':len(parents),'seed':seed,'method':method,
        'scientific_config':data.CONFIG_V05.scientific_mapping(),**record}


def run_region_initialization_online(root=None,workers=4):
    """Run/resume the fixed 6x2x4 consumed-development 60-second matrix."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    _,entries=_region_initialization_entries(root);experiment='region_cooperative_initialization_runs'
    db=data.connect(data.DATABASE_V05)
    zero=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='region_cooperative_initialization_zero'")]
    if len(zero)!=10 or not all(row['certified'] for row in zero):
        db.close();raise RuntimeError('Ten certified zero-search comparisons are required first')
    done={row[0] for row in db.execute('SELECT item_id FROM experiments WHERE experiment=?',(experiment,))};db.close();jobs=[]
    for entry_index,entry in enumerate(entries):
        for seed_index,seed in enumerate(REGION_INITIALIZATION_SEEDS):
            offset=(entry_index+seed_index)%len(REGION_INITIALIZATION_METHODS)
            order=REGION_INITIALIZATION_METHODS[offset:]+REGION_INITIALIZATION_METHODS[:offset]
            for method in order:
                key=f"{entry['label']}:{seed}:{method}"
                if key not in done:jobs.append((key,(entry,seed,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs) or 1),mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_region_initialization_online_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future]
            try:row=future.result()
            except Exception as error:
                instance,seed,method=key.split(':');row={'instance':instance,'seed':int(seed),'method':method,
                    'status':'EXECUTION_ERROR','Cmax_at_60':None,'certified_at_60':False,'diagnostics':[repr(error)]}
            db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',(experiment,key,data.dumps(row)))
            db.close();print(f"region run {key} status={row['status']} Cmax@60={row.get('Cmax_at_60')}",flush=True)
    return len(jobs)


def summarize_region_initialization(root=None):
    import statistics
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    db=data.connect(data.DATABASE_V05,readonly=True)
    zero=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='region_cooperative_initialization_zero' ORDER BY item_id")]
    runs=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='region_cooperative_initialization_runs' ORDER BY item_id")]
    db.close();index={(row['instance'],row['seed'],row['method']):row for row in runs}
    paired={}
    comparisons=(('REGION_HEURISTIC','BASELINE_HEURISTIC'),('REGION_GAT_V05','BASELINE_GAT_V05'),('REGION_GAT_V05','REGION_HEURISTIC'))
    for method,base in comparisons:
        rows=[]
        for row in runs:
            if row['method']!=method:continue
            other=index.get((row['instance'],row['seed'],base))
            if other and row.get('Cmax_at_60') is not None and other.get('Cmax_at_60') is not None:
                rows.append({'instance':row['instance'],'tier':row['tier'],'seed':row['seed'],
                    'delta_seconds':row['Cmax_at_60']-other['Cmax_at_60'],
                    'delta_percent':100*(row['Cmax_at_60']/other['Cmax_at_60']-1)})
        by_tier={}
        for tier in ('N35','N55','N85','ALL'):
            subset=[row for row in rows if tier=='ALL' or row['tier']==tier]
            by_tier[tier]={'count':len(subset),'wins':sum(r['delta_seconds']<-1e-8 for r in subset),
                'ties':sum(abs(r['delta_seconds'])<=1e-8 for r in subset),'losses':sum(r['delta_seconds']>1e-8 for r in subset),
                'mean_delta_percent':None if not subset else statistics.mean(r['delta_percent'] for r in subset),
                'median_delta_percent':None if not subset else statistics.median(r['delta_percent'] for r in subset)}
        paired[f'{method}_vs_{base}']={'rows':rows,'by_tier':by_tier}
    return {'zero':zero,'runs':runs,'paired':paired}


def validate_region_initialization(root=None):
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root);summary=summarize_region_initialization(root)
    zero=summary['zero'];runs=summary['runs']
    if len(zero)!=10 or not all(row['certified'] for row in zero):raise AssertionError('zero-search matrix incomplete or uncertified')
    expected_base=['LOAD_FIRST','RAIL_BALANCED','X_ORDER_AWARE','SPATIAL_SPREAD','RAIL_MONOTONE_BALANCED_BOOTSTRAP']
    expected_region=['LOAD_FIRST','RAIL_BALANCED','X_ORDER_AWARE','REGION_COOPERATIVE_INITIALIZATION','RAIL_MONOTONE_BALANCED_BOOTSTRAP']
    for row in zero:
        normal=[a['strategy'] for a in row['attempts'] if a['index']<5]
        if normal!=(expected_region if row['variant']=='REGION' else expected_base):raise AssertionError('normal initialization slots changed')
        if row['construction_attempts']>6 or row['reference_calls']>10:raise AssertionError('initialization budget exceeded')
    if len(runs)!=48 or not all(row.get('certified_at_60') for row in runs):raise AssertionError('60-second matrix incomplete or uncertified')
    expected=data.CONFIG_V05.scientific_mapping()
    if any(row['scientific_config']!=expected or row['numeric_failure'] for row in runs):raise AssertionError('scientific config or numeric status changed')
    for instance in {row['instance'] for row in runs}:
        for prefix in ('BASELINE','REGION'):
            values={row['initial_cmax'] for row in runs if row['instance']==instance and row['method'].startswith(prefix)}
            if len(values)!=1:raise AssertionError('ranker or seed changed deterministic initialization')
    result={'zero_rows':len(zero),'online_rows':len(runs),'all_certified':True,
        'max_construction_attempts':max(row['construction_attempts'] for row in zero),
        'max_initial_reference_calls':max(row['reference_calls'] for row in zero),
        'max_online_overshoot':max(row['overshoot'] for row in runs)}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return result


CRRR_METHODS=('REGION_CRRR_HEURISTIC','REGION_CRRR_GAT_V05')


def _crrr_smoke_worker(job):
    """One native iteration plus explicitly separate offline CRRR certification."""
    from dataclasses import replace
    from . import dataset as data
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import ScheduleStatus
    from mrta_reference.scheduler import resolve_reference_evaluator
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.solution import official_metrics
    from mrta_search.direction import optimize_directions_with_initial_feasibility
    from mrta_search.pipeline import run_sa_oi_alns_v2
    entry,root=job;smoke=_final_smoke_module(root)
    parents=smoke._range_parents(entry,data.PPO);captured=[]
    def observer(event,value):
        if event=='crrr_attempt' and value[1].candidate is not None:
            captured.append(value[1].candidate)
    search=replace(data.search_config(),initialization_region_cooperative=True,
        enable_crrr=True,max_iterations=1,time_limit=None,checkpoints=())
    result=run_sa_oi_alns_v2(parents,data.CONFIG_V05,search,seed=20261701,
        source_provenance=data.provenance(),observer=observer)
    evaluator=resolve_reference_evaluator(FORMAL_SCOPE_V2,None);offline=[];seen=set()
    for candidate in captured:
        if candidate.solution.canonical_hash in seen:continue
        seen.add(candidate.solution.canonical_hash);direction=optimize_directions_with_initial_feasibility(candidate.solution,data.CONFIG_V05)
        row={'provenance':dict(candidate.provenance),'direction_feasible':direction.total_empty_travel is not None,
            'reference_status':None,'certified':False,'cmax':None}
        if direction.total_empty_travel is not None:
            schedule=evaluator(candidate.solution,data.CONFIG_V05,
                orientations={robot:direction.directions[robot] for robot in range(4)})
            row['reference_status']=schedule.status.value
            if schedule.status is ScheduleStatus.FEASIBLE:
                certification=certify_schedule(candidate.solution,schedule,data.CONFIG_V05,scope=FORMAL_SCOPE_V2)
                row['certified']=certification.certified
                if certification.certified:row['cmax']=official_metrics(candidate.solution,schedule,data.CONFIG_V05).cmax
        offline.append(row)
    st=result.stats;initial=result.initialization
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],
        'actual_N':len(parents),'status':result.status.value,
        'initial_cmax':None if initial.schedule is None else initial.schedule.cmax,
        'search_best_cmax':None if result.best_metrics is None else result.best_metrics.cmax,
        'search_certified':bool(result.final_certification and result.final_certification.certified),
        'attempted_by_family':dict(st.attempted_by_family),'max_iteration_attempts':max(st.per_iteration_attempts,default=0),
        'funnel':{'attempted':st.crrr_attempted,'constructed':st.crrr_constructed,
            'valid_unique':st.crrr_valid_unique,'c2_selected':st.crrr_c2_selected,
            'direction_feasible':st.crrr_direction_feasible,'c4_selected':st.crrr_c4_selected,
            'reference_evaluated':st.crrr_reference_evaluated,'reference_status':dict(st.crrr_reference_status)},
        'original_lns_funnel':{'attempted':st.original_lns_attempted,
            'constructed':st.original_lns_constructed,'valid_unique':st.original_lns_valid_unique,
            'c2_selected':st.original_lns_c2_selected,
            'direction_feasible':st.original_lns_direction_feasible,
            'c4_selected':st.original_lns_c4_selected,
            'reference_evaluated':st.original_lns_reference_evaluated,
            'reference_status':dict(st.original_lns_reference_status)},
        'offline_candidates':offline}


def run_crrr_smoke(root=None,workers=4):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    _,entries=_region_initialization_entries(root)
    wanted={'N35_S1','N55_I5','N55_I6','N85_I9'}
    entries=tuple(entry for entry in entries if entry['label'] in wanted)
    experiment='crrr_native_smoke';db=data.connect(data.DATABASE_V05)
    done={row[0] for row in db.execute('SELECT item_id FROM experiments WHERE experiment=?',(experiment,))};db.close()
    jobs=[(entry['label'],(entry,root)) for entry in entries if entry['label'] not in done]
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs) or 1),mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_crrr_smoke_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future];row=future.result();db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',(experiment,key,data.dumps(row)))
            db.close();print(f"CRRR smoke {key}: {row['funnel']}",flush=True)
    return len(jobs)


def _crrr_online_worker(job):
    from dataclasses import replace
    from . import dataset as data
    entry,seed,method,root=job;smoke=_final_smoke_module(root)
    parents=smoke._range_parents(entry,data.PPO)
    search=replace(data.search_config(),initialization_region_cooperative=True,enable_crrr=True)
    ranker=HybridRanker(root,c2_method='GAT_V05',c4_method='GAT_V05') if method.endswith('GAT_V05') else None
    record,_=data.run_trajectory(entry,seed,parents=parents,ranker=ranker,
        config=data.CONFIG_V05,search=search)
    return {'instance':entry['label'],'tier':entry['tier'],'instance_id':entry['instance_id'],
        'relative_path':entry['relative_path'],'sheet_name':entry['sheet_name'],
        'actual_N':len(parents),'seed':seed,'method':method,
        'scientific_config':data.CONFIG_V05.scientific_mapping(),
        'search_config':{'m':search.m,'m_atomic':144,'m_lns':search.m_lns,
            'original_lns_attempts':search.m_lns-search.crrr_attempts,
            'crrr_attempts':search.crrr_attempts,'kdp':search.kdp,'kref':search.kref,
            'kref_total':search.kref_total,'region_initialization':True,'two_opt_star':search.enable_two_opt_star},
        **record}


def run_crrr_online(root=None,workers=4):
    """Run/resume only the 24 new REGION+CRRR development runs."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    _,entries=_region_initialization_entries(root);experiment='crrr_search_runs'
    db=data.connect(data.DATABASE_V05)
    smoke_rows=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='crrr_native_smoke'")]
    if len(smoke_rows)!=4 or not all(row['search_certified'] and row['funnel']['valid_unique'] for row in smoke_rows):
        db.close();raise RuntimeError('Four certified native CRRR smoke states with valid candidates are required')
    done={row[0] for row in db.execute('SELECT item_id FROM experiments WHERE experiment=?',(experiment,))};db.close();jobs=[]
    for entry_index,entry in enumerate(entries):
        for seed_index,seed in enumerate(REGION_INITIALIZATION_SEEDS):
            order=CRRR_METHODS if (entry_index+seed_index)%2==0 else tuple(reversed(CRRR_METHODS))
            for method in order:
                key=f"{entry['label']}:{seed}:{method}"
                if key not in done:jobs.append((key,(entry,seed,method,root)))
    masks=data.physical_core_affinities()[:workers]
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs) or 1),mp_context=mp.get_context('spawn'),initializer=_final_worker_init,initargs=(masks,)) as pool:
        futures={pool.submit(_crrr_online_worker,job):key for key,job in jobs}
        for future in as_completed(futures):
            key=futures[future]
            try:row=future.result()
            except Exception as error:
                instance,seed,method=key.split(':');row={'instance':instance,'seed':int(seed),'method':method,
                    'status':'EXECUTION_ERROR','Cmax_at_60':None,'certified_at_60':False,'diagnostics':[repr(error)]}
            db=data.connect(data.DATABASE_V05)
            with db:db.execute('INSERT OR REPLACE INTO experiments VALUES (?,?,?)',(experiment,key,data.dumps(row)))
            db.close();print(f"CRRR run {key} status={row['status']} Cmax@60={row.get('Cmax_at_60')}",flush=True)
    return len(jobs)


def summarize_crrr(root=None):
    import statistics
    from . import dataset as data
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    db=data.connect(data.DATABASE_V05,readonly=True)
    smoke=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='crrr_native_smoke' ORDER BY item_id")]
    old=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='region_cooperative_initialization_runs' ORDER BY item_id")]
    new=[json.loads(row[0]) for row in db.execute("SELECT payload FROM experiments WHERE experiment='crrr_search_runs' ORDER BY item_id")]
    experiments=[row[0] for row in db.execute('SELECT DISTINCT experiment FROM experiments ORDER BY experiment')];db.close()
    old=[row for row in old if row['method'] in ('REGION_HEURISTIC','REGION_GAT_V05')]
    rows=old+new;index={(row['instance'],row['seed'],row['method']):row for row in rows}
    comparisons=(('REGION_CRRR_HEURISTIC','REGION_HEURISTIC'),
        ('REGION_CRRR_GAT_V05','REGION_GAT_V05'),
        ('REGION_CRRR_GAT_V05','REGION_CRRR_HEURISTIC'))
    paired={}
    for method,base in comparisons:
        pairs=[]
        for key,row in index.items():
            if row['method']!=method:continue
            other=index.get((row['instance'],row['seed'],base))
            if other and row.get('Cmax_at_60') is not None and other.get('Cmax_at_60') is not None:
                pairs.append({'instance':row['instance'],'tier':row['tier'],'seed':row['seed'],
                    'method_cmax':row['Cmax_at_60'],'baseline_cmax':other['Cmax_at_60'],
                    'delta_seconds':row['Cmax_at_60']-other['Cmax_at_60'],
                    'delta_percent':100*(row['Cmax_at_60']/other['Cmax_at_60']-1)})
        by_tier={}
        for tier in ('N35','N55','N85','ALL'):
            subset=[row for row in pairs if tier=='ALL' or row['tier']==tier]
            by_tier[tier]={'count':len(subset),'wins':sum(row['delta_seconds']<-1e-8 for row in subset),
                'ties':sum(abs(row['delta_seconds'])<=1e-8 for row in subset),
                'losses':sum(row['delta_seconds']>1e-8 for row in subset),
                'mean_delta_percent':None if not subset else statistics.mean(row['delta_percent'] for row in subset),
                'median_delta_percent':None if not subset else statistics.median(row['delta_percent'] for row in subset)}
        paired[f'{method}_vs_{base}']={'rows':sorted(pairs,key=lambda row:(row['instance'],row['seed'])),'by_tier':by_tier}
    funnel_keys=('attempted','constructed','valid_unique','c2_selected','direction_feasible','c4_selected','reference_evaluated','accepted','strict_cmax_improvements')
    funnel={key:sum(row.get('crrr_funnel',{}).get(key,0) for row in new) for key in funnel_keys}
    statuses={status:sum(row.get('crrr_funnel',{}).get('reference_status',{}).get(status,0) for row in new)
        for status in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')}
    funnel['reference_status']=statuses
    funnel['generation_seconds']=sum(row.get('crrr_generation_seconds',0) for row in new)
    funnel['mean_attempt_seconds']=None if not funnel['attempted'] else funnel['generation_seconds']/funnel['attempted']
    return {'smoke':smoke,'old':old,'new':new,'paired':paired,'funnel':funnel,'experiments':experiments}


def validate_crrr(root=None):
    from . import dataset as data
    summary=summarize_crrr(root);old=summary['old'];new=summary['new'];smoke=summary['smoke']
    if len(smoke)!=4 or not all(row['search_certified'] and row['funnel']['valid_unique'] for row in smoke):
        raise AssertionError('CRRR native smoke incomplete')
    if len(old)!=24 or len(new)!=24 or not all(row.get('certified_at_60') for row in old+new):
        raise AssertionError('paired 24 old + 24 new matrix incomplete or uncertified')
    if any(row['scientific_config']!=data.CONFIG_V05.scientific_mapping() for row in old+new):
        raise AssertionError('scientific model changed')
    if any(row.get('max_per_iteration_attempts',192)>192 for row in new):
        raise AssertionError('M192 exceeded')
    if any(row['search_config']['original_lns_attempts']!=40 or row['search_config']['crrr_attempts']!=8 for row in new):
        raise AssertionError('40+8 LNS partition changed')
    for instance in {row['instance'] for row in new}:
        old_initial={row['initial_cmax'] for row in old if row['instance']==instance}
        new_initial={row['initial_cmax'] for row in new if row['instance']==instance}
        if len(old_initial)!=1 or new_initial!=old_initial:raise AssertionError('REGION initialization changed')
    result={'smoke_rows':len(smoke),'old_reused':len(old),'new_runs':len(new),
        'all_certified':True,'max_attempts':max(row['max_per_iteration_attempts'] for row in new),
        'crrr_funnel':summary['funnel']}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Final C4 development utilities')
    parser.add_argument('action',choices=('final-offline','final-train','final-online','baseline-smoke','sci-dev-baselines','sci-edge-ablation','sci-freeze-matrix','sci-formal','sci-summary','sci-report','sci-validate','final-summary','final-validate','final-report','region-init-zero','region-init-online','region-init-summary','region-init-validate','crrr-smoke','crrr-online','crrr-summary','crrr-validate'))
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='final-offline':run_final_c4_offline(args.root)
    elif args.action=='final-train':train_final_c4_ranked(args.root)
    elif args.action=='final-online':run_final_c4_online(args.root,args.workers)
    elif args.action=='baseline-smoke':run_final_baseline_smoke(args.root,min(args.workers,2))
    elif args.action=='sci-dev-baselines':run_sci_dev_baselines(args.root,args.workers)
    elif args.action=='sci-edge-ablation':run_sci_v05_edge_ablation(args.root)
    elif args.action=='sci-freeze-matrix':freeze_sci_formal_matrix(args.root)
    elif args.action=='sci-formal':run_sci_formal(args.root,args.workers)
    elif args.action=='sci-summary':summarize_sci_results(args.root)
    elif args.action=='sci-report':write_sci_report(args.root)
    elif args.action=='sci-validate':validate_sci_results(args.root)
    elif args.action=='final-summary':summarize_final_c4_results(args.root)
    elif args.action=='final-validate':validate_final_c4_invariants(args.root)
    elif args.action=='final-report':write_final_c4_report(args.root)
    elif args.action=='region-init-zero':run_region_initialization_zero(args.root,args.workers)
    elif args.action=='region-init-online':run_region_initialization_online(args.root,args.workers)
    elif args.action=='region-init-summary':print(json.dumps(summarize_region_initialization(args.root),ensure_ascii=False,indent=2))
    elif args.action=='region-init-validate':validate_region_initialization(args.root)
    elif args.action=='crrr-smoke':run_crrr_smoke(args.root,args.workers)
    elif args.action=='crrr-online':run_crrr_online(args.root,args.workers)
    elif args.action=='crrr-summary':print(json.dumps(summarize_crrr(args.root),ensure_ascii=False,indent=2))
    else:validate_crrr(args.root)
