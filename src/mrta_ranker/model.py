"""Small independent legacy and material-ranking two-head MLPs; importing scientific search never imports torch."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class StaticGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    parent_index: dict[str,int]


def build_static_graph(parents,config,*,include_weld_edges=True):
    """Deterministic sparse parent-weld/robot graph from immutable geometry only."""
    from mrta_reference.geometry import build_legal_pattern_catalog,whole_eligible_rails
    from mrta_reference.model import Rail,SplitKind
    from mrta_reference.scope import FORMAL_SCOPE_V2
    parents=tuple(sorted(parents,key=lambda p:p.parent_id));n=len(parents)
    catalog=build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)
    def qualified_rails(parent):
        rails=set(whole_eligible_rails(parent.start,parent.end,config));options=catalog[parent.parent_id]
        if any(q.kind is SplitKind.Y_SPLIT for q in options):rails.update((Rail.UPPER,Rail.LOWER))
        rails.update(q.rail for q in options if q.kind is SplitKind.X_SPLIT)
        return rails
    mid=[]
    for p in parents: mid.append(((p.start[0]+p.end[0])/2,(p.start[1]+p.end[1])/2))
    nearest=[]
    for i in range(n):
        ds=sorted(math.dist(mid[i],mid[j]) for j in range(n) if j!=i)
        nearest.append(sum(ds[:min(4,len(ds))])/max(1,min(4,len(ds))))
    rows=[]
    for i,p in enumerate(parents):
        dx=p.end[0]-p.start[0];dy=p.end[1]-p.start[1];length=p.length
        eligible=qualified_rails(p);options=catalog[p.parent_id]
        rows.append([1.,0.,p.start[0]/20,p.start[1]/12,p.end[0]/20,p.end[1]/12,dx/20,dy/12,length/24,
            dx/length,dy/length,config.process_time(length)/2500,float(Rail.UPPER in eligible),float(Rail.LOWER in eligible),
            float(any(q.kind is SplitKind.Y_SPLIT for q in options)),float(any(q.kind is SplitKind.X_SPLIT for q in options)),nearest[i]/24,i/max(1,n-1)])
    for robot in range(4):
        upper=robot<2;left=robot%2==0
        rows.append([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,float(upper),float(not upper),float(left),float(not left),robot/3,1.])
    edges=[];types=[]
    def edge(a,b,t): edges.append((a,b));types.append(t)
    for i in range(n+4): edge(i,i,0)
    if include_weld_edges:
        for i in range(n):
            close=sorted((math.dist(mid[i],mid[j]),j) for j in range(n) if j!=i)[:4]
            for _,j in close: edge(i,j,1)
        for i in range(n):
            ei=qualified_rails(parents[i])
            xi=(min(parents[i].start[0],parents[i].end[0]),max(parents[i].start[0],parents[i].end[0]))
            related=[]
            for j in range(n):
                if i==j or not ei.intersection(qualified_rails(parents[j])): continue
                xj=(min(parents[j].start[0],parents[j].end[0]),max(parents[j].start[0],parents[j].end[0]))
                gap=max(0.,max(xi[0],xj[0])-min(xi[1],xj[1]))
                if gap<=config.interference_dx: related.append((math.dist(mid[i],mid[j]),j))
            for _,j in sorted(related)[:4]: edge(i,j,2)
    for i,p in enumerate(parents):
        eligible=qualified_rails(p)
        for robot in range(4):
            rail=Rail.UPPER if robot<2 else Rail.LOWER
            if rail in eligible:
                edge(i,n+robot,3);edge(n+robot,i,3)
    return StaticGraph(torch.tensor(rows,dtype=torch.float32),torch.tensor(edges,dtype=torch.long).t().contiguous(),
        torch.tensor(types,dtype=torch.long),{p.parent_id:i for i,p in enumerate(parents)})


class SparseGATLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,heads=4):
        super().__init__()
        if hidden_dim%heads: raise ValueError('hidden dimension must divide heads')
        self.heads=heads;self.width=hidden_dim//heads
        self.linear=nn.Linear(input_dim,hidden_dim,bias=False)
        self.src=nn.Parameter(torch.empty(heads,self.width));self.dst=nn.Parameter(torch.empty(heads,self.width))
        self.relation=nn.Embedding(4,heads);self.bias=nn.Parameter(torch.zeros(hidden_dim));self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight);nn.init.xavier_uniform_(self.src);nn.init.xavier_uniform_(self.dst);nn.init.zeros_(self.relation.weight)
    def forward(self,x,edge_index,edge_type):
        h=self.linear(x).view(x.shape[0],self.heads,self.width);source,destination=edge_index
        logits=(h[source]*self.src).sum(-1)+(h[destination]*self.dst).sum(-1)+self.relation(edge_type)
        logits=nn.functional.leaky_relu(logits,.2);out=torch.zeros_like(h)
        for node in range(x.shape[0]):
            mask=destination==node
            if bool(mask.any()):
                alpha=torch.softmax(logits[mask],dim=0)
                out[node]=(alpha.unsqueeze(-1)*h[source[mask]]).sum(0)
        return out.reshape(x.shape[0],-1)+self.bias


class StaticGATEncoder(nn.Module):
    def __init__(self,input_dim=18,hidden_dim=64,heads=4):
        super().__init__();self.layer1=SparseGATLayer(input_dim,hidden_dim,heads);self.layer2=SparseGATLayer(hidden_dim,hidden_dim,heads)
    def forward(self,graph):
        x=nn.functional.elu(self.layer1(graph.node_features,graph.edge_index,graph.edge_type))
        return self.layer2(x,graph.edge_index,graph.edge_type)


class GraphCandidateRanker(nn.Module):
    """Two-layer four-head GAT plus order-sensitive before/after route fusion."""
    def __init__(self,c2_width=87,c4_width=130,hidden_dim=64,heads=4):
        super().__init__();self.encoder=StaticGATEncoder(18,hidden_dim,heads)
        self.block=nn.Sequential(nn.Linear(hidden_dim+12,32),nn.ReLU())
        self.route=nn.Sequential(nn.Linear(4*2*5*32,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.numeric=nn.ModuleDict({'C2':nn.Sequential(nn.Linear(c2_width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()),
                                    'C4':nn.Sequential(nn.Linear(c4_width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())})
        self.fusion=nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.feasibility=nn.ModuleDict({s:nn.Linear(64,1) for s in ('C2','C4')})
        self.material=nn.ModuleDict({s:nn.Linear(64,1) for s in ('C2','C4')})

    def encode_static(self,graph): return self.encoder(graph)

    def _solution_routes(self,solution,node_embeddings,parent_index,config):
        from mrta_reference.solution import block_map
        from mrta_reference.model import SplitKind
        blocks=block_map(solution,config);patterns={p.parent_id:p for p in solution.patterns};summaries=[]
        for route in solution.routes:
            tokens=[];count=len(route.block_ids)
            for position,block_id in enumerate(route.block_ids):
                b=blocks[block_id];pattern=patterns[b.parent_id];kind=pattern.kind
                extra=node_embeddings.new_tensor([b.start[0]/20,b.start[1]/12,b.end[0]/20,b.end[1]/12,b.u_start,b.u_end,b.length/24,
                    route.robot_id/3,position/max(1,count-1),float(kind is SplitKind.WHOLE),float(kind is SplitKind.Y_SPLIT),float(kind is SplitKind.X_SPLIT)])
                tokens.append(self.block(torch.cat((node_embeddings[parent_index[b.parent_id]],extra))))
            if not tokens:
                summaries.append(node_embeddings.new_zeros(5*32));continue
            seq=torch.stack(tokens);weights=torch.linspace(1/len(tokens),1.,len(tokens),device=seq.device).unsqueeze(1)
            adjacent=(seq[1:]-seq[:-1]).mean(0) if len(tokens)>1 else seq.new_zeros(32)
            summaries.append(torch.cat((seq.mean(0),(seq*weights).sum(0)/weights.sum(),seq[0],seq[-1],adjacent)))
        return torch.cat(summaries)

    def score_candidates(self,stage,graph,node_embeddings,current,candidates,features,mean,std,config):
        before=self._solution_routes(current,node_embeddings,graph.parent_index,config)
        route_rows=[torch.cat((before,self._solution_routes(c.solution,node_embeddings,graph.parent_index,config))) for c in candidates]
        route_hidden=self.route(torch.stack(route_rows))
        x=torch.as_tensor(transform(features,mean,std),dtype=torch.float32,device=node_embeddings.device)
        h=self.fusion(torch.cat((route_hidden,self.numeric[stage](x)),dim=1))
        return self.feasibility[stage](h).squeeze(-1),self.material[stage](h).squeeze(-1)


def save_graph_model(path,model,c2_names,c4_names,scalers,metrics):
    torch.save({'state_dict':model.state_dict(),'c2_names':list(c2_names),'c4_names':list(c4_names),
        'scalers':{k:np.asarray(v).tolist() for k,v in scalers.items()},'metrics':metrics,'architecture':{'layers':2,'heads':4,'hidden':64}},path)


def load_graph_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=GraphCandidateRanker(len(checkpoint['c2_names']),len(checkpoint['c4_names']));model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint


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


def save_v05_mlp(path,model,stage,feature_names,mean,std,metrics):
    """Self-contained 0.5 m checkpoint; no shared scaler side file."""
    torch.save({'state_dict':model.state_dict(),'stage':stage,'width':len(feature_names),'feature_names':list(feature_names),
        'objective':'material','mean':np.asarray(mean).tolist(),'std':np.asarray(std).tolist(),'metrics':metrics,'domain':'0.5m'},path)


def pairwise_loss(positive,negative,weights=None):
    losses=nn.functional.softplus(-(positive-negative))
    if not losses.numel(): return (positive.sum()+negative.sum())*0.0
    return losses.mean() if weights is None else (losses*weights).sum()


def ranking_loss(feasibility,material,y_feasible,y_material,pos_weight,positive,negative,pair_weights=None,pairwise=True):
    feasible_bce=nn.functional.binary_cross_entropy_with_logits(feasibility,y_feasible)
    material_bce=nn.functional.binary_cross_entropy_with_logits(material,y_material,pos_weight=pos_weight)
    pair=pairwise_loss(positive,negative,pair_weights) if pairwise else material.sum()*0.0
    return feasible_bce+material_bce+pair,feasible_bce,material_bce,pair
