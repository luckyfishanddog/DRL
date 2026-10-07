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
    def __init__(self,variant,root=None):
        if variant not in ('MLP-C2','MLP-C4','MLP-BOTH'): raise ValueError('Unknown ranker variant')
        from .model import load_model
        import torch,numpy as np
        torch.set_num_threads(1)
        self.variant=variant;self.inference_seconds=0.0
        root=Path(__file__).resolve().parents[2] if root is None else Path(root)
        self.scalers=np.load(root/'models/mlp_scalers.npz')
        self.models={}
        for stage in ('C2','C4'):
            if variant in ('MLP-'+stage,'MLP-BOTH'): self.models[stage]=load_model(root/f'models/mlp_{stage.lower()}.pt')
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
