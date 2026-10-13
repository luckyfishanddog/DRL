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

    @staticmethod
    def _route_data(solution,config):
        """Return collision-safe route cache keys and their real block geometry."""
        from mrta_reference.solution import block_map
        blocks=block_map(solution,config);patterns={p.parent_id:p for p in solution.patterns};rows=[]
        for route in solution.routes:
            values=tuple((blocks[item].parent_id,blocks[item].block_id,blocks[item].u_start,blocks[item].u_end,
                blocks[item].start,blocks[item].end,patterns[blocks[item].parent_id].kind.value,
                patterns[blocks[item].parent_id].point_id,patterns[blocks[item].parent_id].t) for item in route.block_ids)
            rows.append(((route.robot_id,values),route,tuple(blocks[item] for item in route.block_ids),patterns))
        return rows

    def _batched_solution_routes(self,solutions,node_embeddings,parent_index,config):
        """Encode every unique ordered route once and batch all block projections."""
        route_rows=[];unique={}
        for solution in solutions:
            keys=[]
            for key,route,blocks,patterns in self._route_data(solution,config):
                keys.append(key);unique.setdefault(key,(route,blocks,patterns))
            route_rows.append(tuple(keys))
        inputs=[];spans={}
        for key,(route,blocks,patterns) in unique.items():
            begin=len(inputs);count=len(blocks)
            for position,b in enumerate(blocks):
                kind=patterns[b.parent_id].kind
                extra=node_embeddings.new_tensor([b.start[0]/20,b.start[1]/12,b.end[0]/20,b.end[1]/12,b.u_start,b.u_end,b.length/24,
                    route.robot_id/3,position/max(1,count-1),float(kind.value=='WHOLE'),float(kind.value=='Y_SPLIT'),float(kind.value=='X_SPLIT')])
                inputs.append(torch.cat((node_embeddings[parent_index[b.parent_id]],extra)))
            spans[key]=(begin,len(inputs))
        tokens=self.block(torch.stack(inputs)) if inputs else node_embeddings.new_zeros((0,32));summaries={}
        for key in unique:
            begin,end=spans[key];seq=tokens[begin:end]
            if not len(seq):summaries[key]=node_embeddings.new_zeros(5*32);continue
            weights=torch.linspace(1/len(seq),1.,len(seq),device=seq.device).unsqueeze(1)
            adjacent=(seq[1:]-seq[:-1]).mean(0) if len(seq)>1 else seq.new_zeros(32)
            summaries[key]=torch.cat((seq.mean(0),(seq*weights).sum(0)/weights.sum(),seq[0],seq[-1],adjacent))
        return [torch.cat(tuple(summaries[key] for key in keys)) for keys in route_rows]

    def score_candidates_legacy(self,stage,graph,node_embeddings,current,candidates,features,mean,std,config,**_):
        """Original V05 implementation retained for numerical equivalence checks."""
        before=self._solution_routes(current,node_embeddings,graph.parent_index,config)
        route_rows=[torch.cat((before,self._solution_routes(c.solution,node_embeddings,graph.parent_index,config))) for c in candidates]
        route_hidden=self.route(torch.stack(route_rows))
        x=torch.as_tensor(transform(features,mean,std),dtype=torch.float32,device=node_embeddings.device)
        h=self.fusion(torch.cat((route_hidden,self.numeric[stage](x)),dim=1))
        return self.feasibility[stage](h).squeeze(-1),self.material[stage](h).squeeze(-1)

    def score_candidates(self,stage,graph,node_embeddings,current,candidates,features,mean,std,config,**_):
        encoded=self._batched_solution_routes((current,*(c.solution for c in candidates)),node_embeddings,graph.parent_index,config)
        before=encoded[0];route_rows=[torch.cat((before,after)) for after in encoded[1:]]
        route_hidden=self.route(torch.stack(route_rows))
        x=torch.as_tensor(transform(features,mean,std),dtype=torch.float32,device=node_embeddings.device)
        h=self.fusion(torch.cat((route_hidden,self.numeric[stage](x)),dim=1))
        return self.feasibility[stage](h).squeeze(-1),self.material[stage](h).squeeze(-1)


class GraphCandidateRankerV2(nn.Module):
    """V06: true route transitions and an independently trainable C4 adapter."""
    architecture_version='v06'
    def __init__(self,c2_width=87,c4_width=130,hidden_dim=64,heads=4):
        super().__init__();self.encoder=StaticGATEncoder(18,hidden_dim,heads)
        self.block=nn.Sequential(nn.Linear(hidden_dim+12,32),nn.ReLU())
        self.transition=nn.ModuleDict({s:nn.Sequential(nn.Linear(86,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU()) for s in ('C2','C4')})
        self.route=nn.ModuleDict({s:nn.Sequential(nn.Linear(4*2*5*32,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()) for s in ('C2','C4')})
        self.numeric=nn.ModuleDict({'C2':nn.Sequential(nn.Linear(c2_width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()),
                                    'C4':nn.Sequential(nn.Linear(c4_width,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())})
        self.fusion=nn.ModuleDict({s:nn.Sequential(nn.Linear(128,64),nn.ReLU()) for s in ('C2','C4')})
        self.feasibility=nn.ModuleDict({s:nn.Linear(64,1) for s in ('C2','C4')})
        self.material=nn.ModuleDict({s:nn.Linear(64,1) for s in ('C2','C4')})

    def encode_static(self,graph):return self.encoder(graph)

    @staticmethod
    def _route_data(solution,config):return GraphCandidateRanker._route_data(solution,config)

    def initialize_c4_adapter(self):
        self.transition['C4'].load_state_dict(self.transition['C2'].state_dict())
        self.route['C4'].load_state_dict(self.route['C2'].state_dict())
        self.fusion['C4'].load_state_dict(self.fusion['C2'].state_dict())

    def _encode_solutions(self,stage,solutions,node_embeddings,parent_index,config,directions=None):
        """Batch block tokens; route keys include exact split geometry and C4 directions."""
        directions=[None]*len(solutions) if directions is None else list(directions)
        route_rows=[];unique={}
        for solution,direction in zip(solutions,directions):
            keys=[]
            vectors=None if direction is None else getattr(direction,'directions',direction)
            for key,route,blocks,patterns in self._route_data(solution,config):
                vector=None if vectors is None else tuple(vectors[route.robot_id])
                full=(key,vector if stage=='C4' else None);keys.append(full);unique.setdefault(full,(route,blocks,patterns,vector))
            route_rows.append(tuple(keys))
        block_inputs=[];spans={}
        for key,(route,blocks,patterns,vector) in unique.items():
            begin=len(block_inputs);count=len(blocks)
            for position,b in enumerate(blocks):
                kind=patterns[b.parent_id].kind.value
                extra=node_embeddings.new_tensor([b.start[0]/20,b.start[1]/12,b.end[0]/20,b.end[1]/12,b.u_start,b.u_end,b.length/24,
                    route.robot_id/3,position/max(1,count-1),float(kind=='WHOLE'),float(kind=='Y_SPLIT'),float(kind=='X_SPLIT')])
                block_inputs.append(torch.cat((node_embeddings[parent_index[b.parent_id]],extra)))
            spans[key]=(begin,len(block_inputs))
        token_rows=self.block(torch.stack(block_inputs)) if block_inputs else node_embeddings.new_zeros((0,32));summaries={}
        for key,(route,blocks,patterns,vector) in unique.items():
            begin,end=spans[key];seq=token_rows[begin:end]
            if not len(seq):summaries[key]=node_embeddings.new_zeros(160);continue
            transitions=[]
            for index,(a,b) in enumerate(zip(blocks,blocks[1:])):
                if stage=='C4' and vector is not None and len(vector)==len(blocks):
                    ao,bo=int(vector[index]),int(vector[index+1]);a_exit=a.end if ao==0 else a.start;b_entry=b.start if bo==0 else b.end
                else:
                    pairs=((a.end,b.start),(a.end,b.end),(a.start,b.start),(a.start,b.end));a_exit,b_entry=min(pairs,key=lambda p:(math.dist(*p),p))
                dx=b_entry[0]-a_exit[0];dy=b_entry[1]-a_exit[1]
                geom=[a.start[0]/20,a.start[1]/12,a.end[0]/20,a.end[1]/12,b.start[0]/20,b.start[1]/12,b.end[0]/20,b.end[1]/12,
                    dx/20,dy/12,math.hypot(dx,dy)/24,((b.start[0]+b.end[0]-a.start[0]-a.end[0])/2)/20,
                    ((b.start[1]+b.end[1]-a.start[1]-a.end[1])/2)/12,*[float(route.robot_id==r) for r in range(4)],
                    index/max(1,len(blocks)-2),a.u_start,a.u_end,b.u_start,b.u_end]
                transitions.append(torch.cat((seq[index],seq[index+1],node_embeddings.new_tensor(geom))))
            adjacent=self.transition[stage](torch.stack(transitions)).mean(0) if transitions else seq.new_zeros(32)
            weights=torch.linspace(1/len(seq),1.,len(seq),device=seq.device).unsqueeze(1)
            summaries[key]=torch.cat((seq.mean(0),(seq*weights).sum(0)/weights.sum(),seq[0],seq[-1],adjacent))
        return [torch.cat(tuple(summaries[key] for key in keys)) for keys in route_rows]

    def _solution_routes(self,solution,node_embeddings,parent_index,config,stage='C2',direction=None):
        return self._encode_solutions(stage,[solution],node_embeddings,parent_index,config,[direction])[0]

    def score_candidates(self,stage,graph,node_embeddings,current,candidates,features,mean,std,config,*,directions=None,current_direction=None):
        candidate_directions=[None]*len(candidates) if directions is None else list(directions)
        encoded=self._encode_solutions(stage,(current,*(c.solution for c in candidates)),node_embeddings,graph.parent_index,config,
            (current_direction,*candidate_directions))
        rows=torch.stack([torch.cat((encoded[0],after)) for after in encoded[1:]])
        route_hidden=self.route[stage](rows);x=torch.as_tensor(transform(features,mean,std),dtype=torch.float32,device=node_embeddings.device)
        h=self.fusion[stage](torch.cat((route_hidden,self.numeric[stage](x)),dim=1))
        return self.feasibility[stage](h).squeeze(-1),self.material[stage](h).squeeze(-1)


def save_graph_model(path,model,c2_names,c4_names,scalers,metrics,*,include_weld_edges=True):
    torch.save({'state_dict':model.state_dict(),'c2_names':list(c2_names),'c4_names':list(c4_names),
        'scalers':{k:np.asarray(v).tolist() for k,v in scalers.items()},'metrics':metrics,
        'architecture':{'version':getattr(model,'architecture_version','v05'),'layers':2,'heads':4,'hidden':64,'include_weld_edges':include_weld_edges}},path)


def load_graph_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    cls=GraphCandidateRankerV2 if checkpoint.get('architecture',{}).get('version')=='v06' else GraphCandidateRanker
    model=cls(len(checkpoint['c2_names']),len(checkpoint['c4_names']));model.load_state_dict(checkpoint['state_dict']);model.eval()
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
