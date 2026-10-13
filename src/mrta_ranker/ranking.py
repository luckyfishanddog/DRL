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
        names={'v05':'gat_v05.pt','v06':'gat_v06.pt','v06_no_weld_edges':'gat_v06_no_weld_edges.pt'}
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
