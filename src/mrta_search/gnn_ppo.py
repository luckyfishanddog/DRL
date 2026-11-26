"""One shared GAT Actor-Critic: fixed-atom ownership, then joint job routes.

BC replays complete TRAIN teacher assignments, merged job order and directions.
PPO receives exactly one terminal reward from the existing formal B32 Cmax.
There is no split head, old checkpoint, CEM, VNS or direction postprocessor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import csv
import io
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mrta_ranker.model import SparseGATLayer
from mrta_search.gnn_constructor import _vnd_attention
from mrta_reference.fixed_taskset import expand_routes, job_span
from mrta_reference.model import ScientificConfig
from mrta_reference.solution import canonicalize, block_map
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.scheduler import reference_schedule_formal

NODE_DIM = 33
RELATIONS = ('self','continuous_sibling','geometric_neighbor','eligible_robot','owner','route_successor')
CONFIG = ScientificConfig()
TRAIN_FIELDS = ('stage','update','bc_loss','policy_loss','value_loss','entropy','grad_norm',
                'mean_reward','b32_calls','runtime_s','validation','settings','diagnostic')


@dataclass(frozen=True)
class PPOSettings:
    hidden: int = 64
    heads: int = 4
    bc_epochs: int = 8
    ppo_updates: int = 80
    rollouts_per_update: int = 4
    ppo_epochs: int = 4
    validation_every: int = 10
    learning_rate: float = 3.e-4
    clip: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    gamma: float = 1.
    gae_lambda: float = .95
    gradient_clip: float = .5
    infeasible_reward: float = -10.
    seed: int = 20261110


class TaskGraph:
    """Geometry is built from saved atoms only, never from dynamic patterns."""
    def __init__(self, taskset):
        self.taskset = taskset
        self.tasks = taskset.tasks
        self.n = len(self.tasks)
        self.index = {t.task_id:i for i,t in enumerate(self.tasks)}
        self.base = torch.zeros((self.n+4,NODE_DIM))
        parent_index = {p.parent_id:i for i,p in enumerate(taskset.parents)}
        segment_count = {}
        for t in self.tasks:
            segment_count[t.source_parent_id] = max(segment_count.get(t.source_parent_id,0), t.segment_index+1)
        for i,t in enumerate(self.tasks):
            row = [t.start[0]/20,t.start[1]/12,t.end[0]/20,t.end[1]/12,t.length/25,
                   parent_index[t.source_parent_id]/max(1,len(parent_index)-1),
                   t.segment_index/max(1,segment_count[t.source_parent_id]-1),t.u_start,t.u_end,
                   *[float(r in t.eligible_robots) for r in range(4)],
                   1.,0.,0.,0.,0.,0.,0.,0.,1.,0., *([0.]*10)]
            self.base[i] = torch.tensor(row)
        for r in range(4):
            self.base[self.n+r,22] = 1.
            self.base[self.n+r,23+r] = 1.
        self.scale = max(1.,sum(CONFIG.process_time(p.length) for p in taskset.parents)/4.)
        edges, types = [], []
        def add(a,b,t): edges.append((a,b)); types.append(t)
        for i in range(self.n+4): add(i,i,0)
        centers = np.array([((t.start[0]+t.end[0])/2,(t.start[1]+t.end[1])/2) for t in self.tasks])
        distance = np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1)
        for i,t in enumerate(self.tasks):
            keep = [int(j) for j in np.argsort(distance[i],kind='stable') if j != i][:4]
            for j in keep: add(i,j,2)
            if i+1 < self.n and self.tasks[i+1].source_parent_id == t.source_parent_id:
                add(i,i+1,1); add(i+1,i,1)
            for r in t.eligible_robots:
                add(i,self.n+r,3); add(self.n+r,i,3)
        self.edges = torch.tensor(edges,dtype=torch.long).T.contiguous()
        self.types = torch.tensor(types,dtype=torch.long)

    def state(self):
        return ConstructState(self)


class ConstructState:
    def __init__(self, graph):
        self.graph = graph
        self.owners = [-1]*graph.n
        self.cursor = 0
        self.jobs = None
        self.pending = set()
        self.routes = [[] for _ in range(4)]
        self.directions = [[] for _ in range(4)]
        self.heads = [None]*4
        self.tails = [None]*4

    @property
    def phase(self):
        return 0 if self.cursor < self.graph.n else 1

    def form_jobs(self):
        g = self.graph
        allocated = {r:tuple(t.task_id for i,t in enumerate(g.tasks) if self.owners[i] == r) for r in range(4)}
        merged = canonicalize(g.taskset.parents,g.taskset.patterns,allocated,CONFIG,scope=FORMAL_SCOPE_V2)
        blocks = block_map(merged,CONFIG)
        self.jobs = []
        for route in merged.routes:
            for identifier in route.block_ids:
                parent,lo,hi = job_span(identifier)
                indices = tuple(g.index[f'{parent}::seg_{j:03d}'] for j in range(lo,hi))
                self.jobs.append((identifier,route.robot_id,indices,blocks[identifier]))
        self.pending = set(range(len(self.jobs)))

    def processing_loads(self):
        loads = [0.]*4
        previous_parent, previous_owner = None,-1
        for i,t in enumerate(self.graph.tasks):
            owner = self.owners[i]
            if owner >= 0:
                loads[owner] += t.length/CONFIG.weld_speed
                if t.source_parent_id != previous_parent or owner != previous_owner:
                    loads[owner] += CONFIG.t_pre+CONFIG.t_post
            previous_parent, previous_owner = t.source_parent_id,owner
        return loads

    def graph_inputs(self):
        g = self.graph
        nodes = g.base.clone()
        edges, types = [], []
        for i,r in enumerate(self.owners):
            nodes[i,13:18] = 0.
            nodes[i,13 if r < 0 else 14+r] = 1.
            if r >= 0:
                edges.extend(((i,g.n+r),(g.n+r,i))); types.extend((4,4))
        if self.jobs is not None:
            job_map = {job[0]:job for job in self.jobs}
            for r,route in enumerate(self.routes):
                atom_order = []
                for k,(identifier,bit) in enumerate(zip(route,self.directions[r])):
                    indices = job_map[identifier][2]
                    for i in indices:
                        nodes[i,18] = 1.; nodes[i,19] = k/max(1,len(self.jobs)-1); nodes[i,20] = bit
                    atom_order.extend(reversed(indices) if bit else indices)
                for a,b in zip(atom_order,atom_order[1:]):
                    edges.append((a,b)); types.append(5)
        loads = self.processing_loads()
        for r in range(4):
            nodes[g.n+r,27] = loads[r]/g.scale
            nodes[g.n+r,28] = len(self.routes[r])/max(1,g.n)
            if self.tails[r] is not None:
                nodes[g.n+r,29:31] = torch.tensor([self.tails[r][0]/20,self.tails[r][1]/12])
                nodes[g.n+r,31:33] = torch.tensor([self.heads[r][0]/20,self.heads[r][1]/12])
        if edges:
            return nodes,torch.cat((g.edges,torch.tensor(edges,dtype=torch.long).T),dim=1),torch.cat((g.types,torch.tensor(types)))
        return nodes,g.edges,g.types

    def first_legal(self,r,point):
        for s,other in enumerate(self.heads):
            if other is None: continue
            if r//2 == s//2:
                left,right = (point,other) if r < s else (other,point)
                if left[0]+CONFIG.interference_dx > right[0]+CONFIG.numeric_epsilon:
                    return False
            if abs(point[0]-other[0]) <= CONFIG.interference_dx+CONFIG.numeric_epsilon and abs(point[1]-other[1]) <= CONFIG.interference_dy+CONFIG.numeric_epsilon:
                return False
        return True

    def actions(self):
        if self.phase == 0:
            return [(self.cursor,r,0) for r in self.graph.tasks[self.cursor].eligible_robots]
        if self.jobs is None: self.form_jobs()
        all_actions = [(j,self.jobs[j][1],d) for j in sorted(self.pending) for d in (0,1)]
        legal = []
        for j,r,d in all_actions:
            block = self.jobs[j][3]
            point = block.end if d else block.start
            if self.heads[r] is not None or self.first_legal(r,point): legal.append((j,r,d))
        # Immediate first-position masks need not admit a complete future.
        # Always finish the original full coverage, then let the real B32
        # report any infeasibility; no hidden restart/repair/teacher fallback.
        return legal or all_actions

    def step(self, action):
        j,r,d = action
        if self.phase == 0:
            if j != self.cursor or r not in self.graph.tasks[j].eligible_robots:
                raise ValueError('illegal fixed allocation action')
            self.owners[j] = r; self.cursor += 1
            if self.cursor == self.graph.n: self.form_jobs()
        else:
            if j not in self.pending or self.jobs[j][1] != r or d not in (0,1):
                raise ValueError('illegal joint job/robot/direction action')
            identifier,_,_,block = self.jobs[j]
            self.routes[r].append(identifier); self.directions[r].append(d); self.pending.remove(j)
            entry,exit = (block.end,block.start) if d else (block.start,block.end)
            if self.heads[r] is None: self.heads[r] = entry
            self.tails[r] = exit

    @property
    def done(self):
        return self.jobs is not None and not self.pending

    def solution(self):
        if not self.done: raise ValueError('incomplete fixed construction')
        return canonicalize(self.graph.taskset.parents,self.graph.taskset.patterns,
                            dict(enumerate(self.routes)),CONFIG,scope=FORMAL_SCOPE_V2)


class FixedActorCritic(nn.Module):
    def __init__(self, settings=PPOSettings()):
        super().__init__()
        h = settings.hidden
        self.layer1 = SparseGATLayer(NODE_DIM,h,settings.heads)
        self.layer2 = SparseGATLayer(h,h,settings.heads)
        for layer in (self.layer1,self.layer2):
            layer.relation = nn.Embedding(len(RELATIONS),settings.heads)
            nn.init.zeros_(layer.relation.weight)
        # One actor scores both phases; direction is a feature of the complete
        # route action, never an independently optimized module.
        self.actor = nn.Sequential(nn.Linear(3*h+12,h),nn.Tanh(),nn.Linear(h,1))
        self.critic = nn.Sequential(nn.Linear(5*h+3,h),nn.Tanh(),nn.Linear(h,1))

    def forward(self,state):
        nodes,edges,types = state.graph_inputs()
        h = torch.nn.functional.elu(_vnd_attention(self.layer1,nodes,edges,types))
        h = _vnd_attention(self.layer2,h,edges,types)
        g = state.graph; pooled = h[:g.n].mean(0)
        actions = state.actions(); rows = []
        loads = state.processing_loads()
        for j,r,d in actions:
            if state.phase == 0:
                t = g.tasks[j]; embedding = h[j]
                length,u,v,count,entry = t.length,t.u_start,t.u_end,1,t.start
            else:
                _,_,indices,block = state.jobs[j]
                embedding = h[list(indices)].mean(0)
                length,u,v,count = block.length,block.u_start,block.u_end,len(indices)
                entry = block.end if d else block.start
            distance = 0. if state.tails[r] is None else math.dist(state.tails[r],entry)/CONFIG.empty_speed/g.scale
            extra = h.new_tensor([float(state.phase),float(d==0 and state.phase==1),float(d==1),distance,
                                 length/25,u,v,count/max(1,g.n),len(state.routes[r])/max(1,g.n),
                                 loads[r]/g.scale,(len(state.pending) if state.jobs else g.n-state.cursor)/max(1,g.n),
                                 state.cursor/max(1,g.n)])
            rows.append(torch.cat((embedding,h[g.n+r],pooled,extra)))
        logits = self.actor(torch.stack(rows)).squeeze(-1)
        progress = h.new_tensor([float(state.phase),state.cursor/max(1,g.n),
                                sum(map(len,state.routes))/max(1,g.n)])
        value = self.critic(torch.cat((pooled,h[g.n:].flatten(),progress))).squeeze()
        return actions,Categorical(logits=logits),value


def teacher_actions(graph,payload):
    state = graph.state(); actions = []
    for i,t in enumerate(graph.tasks):
        action = (i,int(payload['assignments'][t.task_id]),0)
        actions.append(action); state.step(action)
    ids = {job[0]:j for j,job in enumerate(state.jobs)}
    for k in range(max(map(len,payload['routes']),default=0)):
        for r in range(4):
            if k < len(payload['routes'][r]):
                action = (ids[payload['routes'][r][k]],r,int(payload['directions'][r][k]))
                if action not in state.actions():
                    raise ValueError('B32-feasible teacher trajectory excluded by action mask')
                actions.append(action); state.step(action)
    if not state.done or len(actions) != graph.n+len(state.jobs):
        raise ValueError('teacher trajectory lacks route/direction coverage')
    return actions


def formal_episode(model,graph,seed,*,greedy=False):
    generator = torch.Generator().manual_seed(seed)
    state = graph.state(); actions,logs,values = [],[],[]
    model.eval()
    with torch.no_grad():
        while not state.done:
            available,distribution,value = model(state)
            index = int(torch.argmax(distribution.logits)) if greedy else int(torch.multinomial(distribution.probs,1,generator=generator))
            action = available[index]
            actions.append(action); logs.append(float(distribution.log_prob(torch.tensor(index)))); values.append(float(value))
            state.step(action)
    solution = state.solution()
    directions = tuple(tuple(bits) for bits in state.directions)
    schedule = reference_schedule_formal(solution,CONFIG,scope=FORMAL_SCOPE_V2,orientations=dict(enumerate(directions)))
    return dict(graph=graph,actions=actions,old_log=logs,old_value=values,solution=solution,
                directions=directions,schedule=schedule)


def terminal_reward(schedule,reference,settings=PPOSettings()):
    if not math.isfinite(reference) or reference <= 0.:
        raise ValueError('TRAIN reference Cmax must be positive and fixed')
    return -schedule.cmax/reference if schedule.feasible else settings.infeasible_reward


def generalized_advantages(values,reward,settings=PPOSettings()):
    values = np.asarray(values,dtype=float)
    advantages = np.zeros_like(values); last = 0.
    for i in range(len(values)-1,-1,-1):
        future = values[i+1] if i+1 < len(values) else 0.
        immediate = reward if i == len(values)-1 else 0.
        delta = immediate+settings.gamma*future-values[i]
        last = delta+settings.gamma*settings.gae_lambda*last
        advantages[i] = last
    return advantages,advantages+values


def append_training(api,row):
    path = api.RESULTS/'gnn_ppo_training.csv'
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer,fieldnames=TRAIN_FIELDS)
    if not path.exists(): writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if (path.stat().st_size if path.exists() else 0)+len(text.encode('utf-8')) >= api.LIMIT:
        raise ValueError('training CSV would exceed 100MB')
    with path.open('a',encoding='utf-8',newline='') as stream: stream.write(text); stream.flush()


def save_weights(api,path,model,settings,**metadata):
    buffer = io.BytesIO()
    torch.save(dict(model_state=model.state_dict(),settings=asdict(settings),**metadata),buffer)
    data = buffer.getvalue()
    if len(data) >= api.LIMIT: raise ValueError('model would exceed 100MB')
    path.write_bytes(data)


def validation_score(model,graphs,api):
    rows = []
    for name,graph in graphs:
        for seed in api.SEEDS:
            episode = formal_episode(model,graph,seed)
            s = episode['schedule']
            rows.append(dict(layout_id=name,seed=seed,status=s.status.value,cmax_s=s.cmax if s.feasible else None))
    feasible = [r['cmax_s'] for r in rows if r['status'] == 'FEASIBLE']
    score = (len(rows)-len(feasible),sum(feasible)/len(feasible) if feasible else math.inf)
    return score,rows


def train_experiment(api,settings=PPOSettings()):
    torch.set_num_threads(1)
    random.seed(settings.seed); np.random.seed(settings.seed); torch.manual_seed(settings.seed)
    bc_path = api.ROOT/'models/gnn_fixed_euclidean_bc.pt'
    ppo_path = api.ROOT/'models/gnn_fixed_euclidean_ppo.pt'
    if ppo_path.exists():
        checkpoint = torch.load(ppo_path,map_location='cpu',weights_only=False)
        if checkpoint.get('training_complete'):
            print('FIXED_TRAINING_ALREADY_COMPLETE',flush=True); return
        raise RuntimeError('incomplete saved PPO stage; inspect training CSV before restarting')
    data = api.load_tasksets()
    train = [(name,TaskGraph(ts)) for split,name,ts in data if split == 'train']
    validation = [(name,TaskGraph(ts)) for split,name,ts in data if split == 'validation']
    teachers = {}
    # Read only complete new fixed-domain baseline records as training labels.
    for method in ('wag_vns','hga'):
        rows = api.read_rows(method)
        expected = {(split,name,seed) for split,name,_ in data for seed in api.SEEDS}
        required = 500 if method == 'wag_vns' else 200000
        completed = {(r['split'],r['layout_id'],int(r['seed'])) for r in rows
                     if r['termination_reason'] == 'ITERATION_LIMIT' and int(r['iterations']) == required}
        if completed != expected:
            raise RuntimeError(f'{method}: stage 2 is incomplete ({len(completed)}/{len(expected)}); no training was performed')
        for row in rows:
            if row['split'] != 'train' or row['status'] != 'FEASIBLE': continue
            name = row['layout_id']; old = teachers.get(name)
            if old is None or float(row['cmax_s']) < float(old['cmax_s']): teachers[name] = row
    if not teachers: raise RuntimeError('no B32-feasible TRAIN teacher; retain failed instances and diagnose')
    references = {name:float(teachers[name]['cmax_s']) if name in teachers else graph.scale for name,graph in train}
    examples = [(name,g,teacher_actions(g,json.loads(teachers[name]['solution']))) for name,g in train if name in teachers]
    model = FixedActorCritic(settings)
    optimizer = torch.optim.Adam(model.parameters(),lr=settings.learning_rate)
    metadata = dict(teacher_layout_count=len(examples),train_layout_count=len(train),
                    teacher_source='best B32-feasible WAG/HGA TRAIN record per layout',
                    references=references,reference_sources={name:'B32 teacher Cmax' if name in teachers
                        else 'processing Cmax lower bound; no feasible BC label' for name,_ in train},
                    random_initialization=True)
    print('FIXED_BC_START',api.json_text(dict(settings=asdict(settings),**metadata)),flush=True)
    best_score, best_state, best_epoch = (math.inf,math.inf),None,0
    for epoch in range(1,settings.bc_epochs+1):
        started = time.perf_counter(); order = list(examples); random.shuffle(order)
        losses, value_losses, norms = [],[],[]
        model.train()
        for name,graph,actions in order:
            state = graph.state(); optimizer.zero_grad()
            target = -float(teachers[name]['cmax_s'])/references[name]
            for action in actions:
                available,distribution,value = model(state)
                index = available.index(action)
                bc = -distribution.log_prob(torch.tensor(index)); value_loss = (value-target).square()
                ((bc+.1*value_loss)/len(actions)).backward()
                losses.append(float(bc.detach())); value_losses.append(float(value_loss.detach()))
                state.step(action)
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings.gradient_clip)))
            optimizer.step()
        checked = epoch % 4 == 0 or epoch == settings.bc_epochs
        val_rows = []
        if checked:
            score,val_rows = validation_score(model,validation,api)
            if score < best_score:
                best_score,best_state,best_epoch = score,copy.deepcopy(model.state_dict()),epoch
                save_weights(api,bc_path,model,settings,stage='BC',selected_epoch=epoch,training_complete=False,**metadata)
        row = dict(stage='BC',update=epoch,bc_loss=float(np.mean(losses)),value_loss=float(np.mean(value_losses)),
                   grad_norm=float(np.mean(norms)),b32_calls=len(val_rows),runtime_s=time.perf_counter()-started,
                   validation=api.json_text(val_rows),settings=api.json_text(asdict(settings)),diagnostic=api.json_text(metadata))
        append_training(api,row); print('FIXED_BC_UPDATE',api.json_text(row),flush=True)
    model.load_state_dict(best_state)
    save_weights(api,bc_path,model,settings,stage='BC',selected_epoch=best_epoch,completed_epochs=settings.bc_epochs,
                 training_complete=True,**metadata)
    optimizer = torch.optim.Adam(model.parameters(),lr=settings.learning_rate)
    ppo_best_score,ppo_best_state,selected_update = best_score,copy.deepcopy(model.state_dict()),0
    for update in range(1,settings.ppo_updates+1):
        started = time.perf_counter(); episodes = []
        # Cyclic shuffled TRAIN traversal includes every layout, even one with
        # no feasible teacher; missing BC labels never remove a PPO instance.
        if (update-1)*settings.rollouts_per_update % len(train) == 0:
            random.shuffle(train)
        for k in range(settings.rollouts_per_update):
            index = ((update-1)*settings.rollouts_per_update+k) % len(train)
            name,graph = train[index]
            episode = formal_episode(model,graph,settings.seed+100000*update+k)
            reward = terminal_reward(episode['schedule'],references[name],settings)
            adv,returns = generalized_advantages(episode['old_value'],reward,settings)
            episode.update(reward=reward,advantages=adv,returns=returns); episodes.append(episode)
        all_adv = np.concatenate([e['advantages'] for e in episodes])
        a_mean,a_std = float(all_adv.mean()),max(1.e-8,float(all_adv.std()))
        total = len(all_adv); policies,values,entropies,norms = [],[],[],[]
        for _ in range(settings.ppo_epochs):
            optimizer.zero_grad(); model.train()
            for episode in episodes:
                state = episode['graph'].state()
                for t,action in enumerate(episode['actions']):
                    available,distribution,value = model(state)
                    log = distribution.log_prob(torch.tensor(available.index(action)))
                    ratio = torch.exp(log-episode['old_log'][t])
                    advantage = (episode['advantages'][t]-a_mean)/a_std
                    policy_loss = -torch.minimum(ratio*advantage,ratio.clamp(1-settings.clip,1+settings.clip)*advantage)
                    value_loss = (value-float(episode['returns'][t])).square()
                    entropy = distribution.entropy()
                    ((policy_loss+settings.value_coefficient*value_loss-settings.entropy_coefficient*entropy)/total).backward()
                    policies.append(float(policy_loss.detach())); values.append(float(value_loss.detach())); entropies.append(float(entropy.detach()))
                    state.step(action)
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings.gradient_clip)))
            optimizer.step()
        val_rows = []
        if update % settings.validation_every == 0 or update == settings.ppo_updates:
            score,val_rows = validation_score(model,validation,api)
            if score < ppo_best_score:
                ppo_best_score,ppo_best_state,selected_update = score,copy.deepcopy(model.state_dict()),update
        # Store selected weights in the one approved stage file; log actual
        # optimization separately so model selection cannot erase PPO history.
        selected = FixedActorCritic(settings); selected.load_state_dict(ppo_best_state)
        save_weights(api,ppo_path,selected,settings,stage='PPO',completed_updates=update,
                     selected_update=selected_update,selected_bc_epoch=best_epoch,
                     last_trained_state=model.state_dict(),training_complete=update == settings.ppo_updates,**metadata)
        row = dict(stage='PPO',update=update,policy_loss=float(np.mean(policies)),value_loss=float(np.mean(values)),
                   entropy=float(np.mean(entropies)),grad_norm=float(np.mean(norms)),mean_reward=float(np.mean([e['reward'] for e in episodes])),
                   b32_calls=len(episodes)+len(val_rows),runtime_s=time.perf_counter()-started,validation=api.json_text(val_rows),
                   settings=api.json_text(asdict(settings)),diagnostic=api.json_text(dict(feasible_rollouts=sum(e['schedule'].feasible for e in episodes),
                         selected_validation_failures=ppo_best_score[0],selected_validation_mean_cmax=None if not math.isfinite(ppo_best_score[1]) else ppo_best_score[1])))
        append_training(api,row); print('FIXED_PPO_UPDATE',api.json_text(row),flush=True)
        api.update_report()
    print('FIXED_GNN_TRAINING_COMPLETE',flush=True)


def compare_experiment(api):
    torch.set_num_threads(1)
    path = api.ROOT/'models/gnn_fixed_euclidean_ppo.pt'
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if not checkpoint.get('training_complete'):
        raise ValueError('PPO training has not finished; TEST is not used for selection')
    settings = PPOSettings(**checkpoint['settings'])
    model = FixedActorCritic(settings); model.load_state_dict(checkpoint['model_state']); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    rows = api.read_rows('gnn_ppo')
    done = {(r['split'],r['layout_id'],int(r['seed'])) for r in rows}
    for split,name,ts in api.load_tasksets():
        graph = TaskGraph(ts)
        for seed in api.SEEDS:
            if (split,name,seed) in done: continue
            started = time.perf_counter()
            episode = formal_episode(model,graph,seed)
            schedule = episode['schedule']
            row = dict(split=split,layout_id=name,seed=seed,method='gnn_ppo',status=schedule.status.value,
                       cmax_s=schedule.cmax if schedule.feasible else None,runtime_s=time.perf_counter()-started,
                       iterations=len(episode['actions']),termination_reason='COMPLETE_CONSTRUCTION',b32_calls=1,
                       parameters=api.json_text(asdict(settings)),diagnostic=api.json_text(dict(diagnostics=schedule.diagnostics,
                       selected_ppo_update=checkpoint.get('selected_update'),selected_bc_epoch=checkpoint.get('selected_bc_epoch'),
                       inference='one joint policy construction per seed; frozen weights; original B32')),
                       solution=api.json_text(api.solution_record(episode['solution'],episode['directions'])))
            api.append_row('gnn_ppo',row); api.update_report()
            print('FIXED_GNN_RESULT',api.json_text({k:v for k,v in row.items() if k not in ('solution','diagnostic')}),flush=True)
