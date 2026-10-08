"""Small independent legacy and material-ranking two-head MLPs; importing scientific search never imports torch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn


class CandidateMLP(nn.Module):
    def __init__(self,width,objective="legacy"):
        if objective not in ("legacy","material"): raise ValueError("Unknown objective")
        super().__init__()
        self.embedding=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.objective=objective
        self.feasibility=nn.Linear(64,1)
        if objective=="legacy": self.improvement=nn.Linear(64,1)
        else: self.material=nn.Linear(64,1)
    def forward(self,x):
        h=self.embedding(x)
        return self.feasibility(h).squeeze(-1),(self.improvement(h) if self.objective=="legacy" else self.material(h)).squeeze(-1)


def fit_scaler(train_features):
    x=np.asarray(train_features,dtype=np.float32).astype(np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all(): raise ValueError('Finite TRAIN feature matrix required')
    mean=x.mean(axis=0);std=x.std(axis=0);std[std==0]=1.0
    return mean,std


def transform(features,mean,std):
    # SQLite stores raw features as float32; use identical quantization online.
    x=(np.asarray(features,dtype=np.float32).astype(np.float64)-mean)/std
    if not np.isfinite(x).all(): raise ValueError('Nonfinite standardized features')
    return x.astype(np.float32)


def predict_scores(model,features,mean,std):
    model.eval()
    with torch.inference_mode():
        logit,gain=model(torch.from_numpy(transform(features,mean,std)))
        return (torch.sigmoid(logit)*gain if model.objective=="legacy" else torch.sigmoid(gain)).numpy()


def multitask_loss(logit,gain,y_feasible,y_improvement):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y_feasible)
    mask=y_feasible==1
    regression=nn.functional.smooth_l1_loss(gain[mask],y_improvement[mask]) if mask.any() else gain.sum()*0.0
    return bce+regression,bce,regression


def save_model(path,model,stage,feature_names,epoch,metrics):
    torch.save({'state_dict':model.state_dict(),'stage':stage,'width':len(feature_names),'feature_names':feature_names,
        'epoch':epoch,'dev_selection':metrics,'objective':model.objective},path)


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=CandidateMLP(checkpoint['width'],checkpoint.get('objective','legacy'));model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint


def pairwise_loss(positive,negative,weights=None):
    losses=nn.functional.softplus(-(positive-negative))
    if not losses.numel(): return (positive.sum()+negative.sum())*0.0
    return losses.mean() if weights is None else (losses*weights).sum()


def ranking_loss(feasibility,material,y_feasible,y_material,pos_weight,positive,negative,pair_weights=None,pairwise=True):
    feasible_bce=nn.functional.binary_cross_entropy_with_logits(feasibility,y_feasible)
    material_bce=nn.functional.binary_cross_entropy_with_logits(material,y_material,pos_weight=pos_weight)
    pair=pairwise_loss(positive,negative,pair_weights) if pairwise else material.sum()*0.0
    return feasible_bce+material_bce+pair,feasible_bce,material_bce,pair
