"""Small independent two-head MLPs; importing scientific search never imports torch."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn


class CandidateMLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.embedding=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.feasibility=nn.Linear(64,1);self.improvement=nn.Linear(64,1)
    def forward(self,x):
        h=self.embedding(x)
        return self.feasibility(h).squeeze(-1),self.improvement(h).squeeze(-1)


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
        return (torch.sigmoid(logit)*gain).numpy()


def multitask_loss(logit,gain,y_feasible,y_improvement):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y_feasible)
    mask=y_feasible==1
    regression=nn.functional.smooth_l1_loss(gain[mask],y_improvement[mask]) if mask.any() else gain.sum()*0.0
    return bce+regression,bce,regression


def save_model(path,model,stage,feature_names,epoch,metrics):
    torch.save({'state_dict':model.state_dict(),'stage':stage,'width':len(feature_names),'feature_names':feature_names,
        'epoch':epoch,'dev_selection':metrics},path)


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=CandidateMLP(checkpoint['width']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint
