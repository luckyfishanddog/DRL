from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
import itertools
import math
import random
import time
from typing import Mapping, Sequence

from mrta_reference.geometry import (
    _axis_band_interval,
    finite_x_split_validator,
    oriented_endpoints,
    robot_is_eligible,
)
from mrta_reference.model import (
    CandidateKey,
    CandidateMove,
    CanonicalSolution,
    Operation,
    OperationKind,
    MoveType,
    ParentWeld,
    RobotRoute,
    Route,
    ScheduleResult,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.scheduler import build_operation_templates
from mrta_reference.solution import block_map, canonicalize

from .direction import DirectionVectors, _initial_error
from .initialization import bounded_insertion_positions


@dataclass
class BottleneckAction:
    solution: CanonicalSolution
    directions: tuple
    state: object
    operator: str
    parents: tuple
    signature: tuple
    route_delta_s: tuple
    transfers: dict
    score: float = 0.


def actual_parent_transfers(before, after, config, *, before_blocks=None, after_blocks=None,
                            parents=None):
    """Positive geometric weld length changing owner, including changed cuts."""
    old = before_blocks if before_blocks is not None else block_map(before.solution, config)
    new = after_blocks if after_blocks is not None else block_map(after.solution, config)
    old_owner = {b:r for r,route in enumerate(before.solution.routes) for b in route.block_ids}
    new_owner = {b:r for r,route in enumerate(after.solution.routes) for b in route.block_ids}
    grouped = {}
    for child in new.values():
        grouped.setdefault(child.parent_id, []).append(child)
    changes = {}
    for block in old.values():
        if parents is not None and block.parent_id not in parents:
            continue
        axis = int(abs(block.end[1]-block.start[1]) > abs(block.end[0]-block.start[0]))
        lo,hi = sorted((block.start[axis],block.end[axis]))
        for child in grouped.get(block.parent_id, ()):
            if old_owner[block.block_id] == new_owner[child.block_id]:
                continue
            lower,upper = sorted((child.start[axis],child.end[axis]))
            overlap = min(hi,upper)-max(lo,lower)
            if overlap > config.numeric_epsilon:
                changes.setdefault(block.parent_id, []).append(dict(source=old_owner[block.block_id],
                    destination=new_owner[child.block_id], length_m=overlap/(hi-lo)*block.length))
    return changes


class BottleneckWorkspace:
    """Parent-atomic proposals and reusable route costs; no scheduler or GNN here."""
    def __init__(self, parents, config, scope, space, *, seed=0, candidate_limit=192):
        self.parents,self.config,self.scope,self.space = tuple(parents),config,scope,space
        self.rng = random.Random(seed)
        self.ranking_rng = random.Random(seed+1000003)
        self.candidate_limit = candidate_limit
        self.counts = {name:Counter() for name in ('initial','relocate','swap','two_opt','lns')}
        self.route_cache,self.block_cache,self.raw_seen = {},{},set()
        self.evaluated = set()
        self.route_cache_hits = self.route_cache_builds = 0
        self.actions_by_parent = {}
        for index,assignment in enumerate(space.actions):
            self.actions_by_parent.setdefault(assignment.parent_id, []).append(index)

    @staticmethod
    def identity(solution, directions):
        return solution.patterns,tuple((tuple(r.block_ids),tuple(d)) for r,d in zip(solution.routes,directions))

    def blocks(self, patterns):
        key = tuple(patterns)
        if key not in self.block_cache:
            self.block_cache[key] = {b.block_id:b for p in key for b in self.space.children[p]}
        return self.block_cache[key]

    def state(self, routes, directions, blocks):
        loads,arcs = [],[]
        for route,bits in zip(routes,directions):
            key = tuple((blocks[b].start,blocks[b].end,d) for b,d in zip(route,bits))
            if key in self.route_cache:
                self.route_cache_hits += 1
                p,m = self.route_cache[key]
            else:
                self.route_cache_builds += 1
                p = sum(self.config.process_time(blocks[b].length) for b in route)
                m = _route_arc_costs(route,bits,blocks,self.config)
                self.route_cache[key] = p,m
            loads.append(p);arcs.append(m)
        return OrientedRouteState(tuple(map(tuple,routes)),tuple(map(tuple,directions)),tuple(loads),tuple(arcs))

    @staticmethod
    def cycles(state):
        return tuple(p+sum(m) for p,m in zip(state.loads,state.arc_costs))

    def ownership_structure(self, candidate):
        selected = {p.parent_id:p for p in candidate.solution.patterns}
        owners = {b:r for r,route in enumerate(candidate.state.routes) for b in route}
        return tuple((parent,selected[parent].pattern_id,
                      tuple(owners[b.block_id] for b in self.space.children[selected[parent]]))
                     for parent in candidate.parents)

    def bottleneck(self, current, schedule):
        cycles = self.cycles(current.state)
        # Actual native completion includes waiting; failed native timelines do
        # not supply completion labels. Exact route periods remain observable.
        completion = schedule.robot_completion if schedule.feasible else cycles
        critical = max(range(4), key=lambda r:(completion[r],cycles[r],-r))
        receivers = sorted((r for r in range(4) if r != critical),
                           key=lambda r:(completion[r],cycles[r],r))
        blocks = self.blocks(current.solution.patterns)
        contributions = {}
        for k,b in enumerate(current.state.routes[critical]):
            parent = blocks[b].parent_id
            contribution = self.config.process_time(blocks[b].length)
            if k:contribution += current.state.arc_costs[critical][k-1]
            if k < len(current.state.arc_costs[critical]):contribution += current.state.arc_costs[critical][k]
            contributions[parent] = contributions.get(parent,0.)+contribution
        ordered = sorted(contributions,key=lambda p:(-contributions[p],p))
        return critical,tuple(receivers),tuple(ordered)

    def partial(self, current, removed):
        blocks = self.blocks(current.solution.patterns)
        routes,bits = [],[]
        for route,vector in zip(current.state.routes,current.directions):
            keep = [k for k,b in enumerate(route) if blocks[b].parent_id not in removed]
            routes.append(tuple(route[k] for k in keep));bits.append(tuple(vector[k] for k in keep))
        return self.state(routes,bits,blocks)

    def _positions(self, state, child, robot, blocks, *, limit):
        route,bits = state.routes[robot],state.directions[robot]
        def delta(k):
            best = math.inf
            for d in (0,1):
                start,end = oriented_endpoints(child,d)
                incoming = math.dist(oriented_endpoints(blocks[route[k-1]],bits[k-1])[1],start) if k else 0.
                outgoing = math.dist(end,oriented_endpoints(blocks[route[k]],bits[k])[0]) if k<len(route) else 0.
                removed = (math.dist(oriented_endpoints(blocks[route[k-1]],bits[k-1])[1],
                                     oriented_endpoints(blocks[route[k]],bits[k])[0]) if 0<k<len(route) else 0.)
                best = min(best,incoming+outgoing-removed)
            return best
        # Local insertion geometry proposes positions; only native Cmax accepts
        # a complete action. No weighted processing/MOVE/WAIT objective is used.
        nearest = sorted(range(len(route)+1),key=lambda k:(delta(k),k))[:2]
        spread = list(bounded_insertion_positions(len(route),max(2,limit)))
        self.rng.shuffle(spread)
        return tuple(dict.fromkeys((*nearest,*spread)))[:limit]

    def insert_parent(self, state, assignment, blocks, *, positions=3, keep=6):
        children = self.space.children[assignment.pattern]
        blocks = {**{k:b for k,b in blocks.items() if b.parent_id!=assignment.parent_id},
                  **{b.block_id:b for b in children}}
        beam = [(state,())]
        for child,robot in zip(children,assignment.robots):
            expanded = []
            for partial,actions in beam:
                for k in self._positions(partial,child,robot,blocks,limit=positions):
                    for direction in (0,1):
                        route = partial.routes[robot][:k]+(child.block_id,)+partial.routes[robot][k:]
                        bits = partial.directions[robot][:k]+(direction,)+partial.directions[robot][k:]
                        costs = _insert_route_arc_costs(partial.routes[robot],partial.directions[robot],
                            partial.arc_costs[robot],child,k,direction,blocks,self.config)
                        after = OrientedRouteState(partial.routes[:robot]+(route,)+partial.routes[robot+1:],
                            partial.directions[:robot]+(bits,)+partial.directions[robot+1:],
                            tuple(p+self.config.process_time(child.length)*(r==robot) for r,p in enumerate(partial.loads)),
                            partial.arc_costs[:robot]+(costs,)+partial.arc_costs[robot+1:])
                        expanded.append((after,actions+(OrientedInsertionAction(child.block_id,robot,k,direction),)))
            # Keep both directions and spread positions before a neural or
            # native score exists. This is a bounded proposal beam, not LB ranking.
            if len(expanded)>keep:
                slots = tuple(round(i*(len(expanded)-1)/max(1,keep-1)) for i in range(keep))
                expanded = [expanded[i] for i in dict.fromkeys(slots)]
            beam = expanded
        return tuple((s,a,blocks) for s,a in beam)

    def _transfer_assignments(self, current, parent, source, target, selected):
        owners = {b:r for r,route in enumerate(current.state.routes) for b in route}
        before = tuple(owners[b.block_id] for b in self.space.children[selected])
        transferred = tuple(target if r==source else r for r in before)
        indices = []
        for i in self.actions_by_parent[parent]:
            assignment = self.space.actions[i]
            keep_split = assignment.pattern==selected and assignment.robots==transferred and transferred!=before
            whole_option = (assignment.pattern.kind is SplitKind.WHOLE and assignment.pattern!=selected
                            and assignment.robots==(target,))
            if keep_split or whole_option:indices.append(i)
        return tuple(sorted(indices,key=lambda i:(self.space.actions[i].pattern!=selected,i)))

    def finish(self, current, patterns, state, operator, parents):
        counts = self.counts[operator];counts['generated'] += 1
        patterns = tuple(sorted(patterns,key=lambda p:p.parent_id))
        raw = patterns,state.routes,state.directions
        local = self.raw_seen
        if raw in local or raw == (current.solution.patterns,current.state.routes,current.directions):
            counts['duplicate'] += 1;return None
        local.add(raw)
        blocks = self.blocks(patterns)
        first_points = {r:oriented_endpoints(blocks[route[0]],state.directions[r][0])[0]
                        for r,route in enumerate(state.routes) if route}
        # This is the existing B32 input geometry rule, not a native DEADLOCK
        # filter. Apply it only after both source/target routes are complete.
        if _initial_error(first_points,self.config) is not None:
            counts['invalid'] += 1;counts['initial_geometry_invalid'] += 1;return None
        try:
            solution = canonicalize(self.parents,patterns,dict(enumerate(state.routes)),self.config,scope=self.scope)
            # Do not reuse a noncanonical split's processing/MOVE as a bound
            # after canonicalize merged it. Legal WHOLE proposals are separate.
            if solution.patterns!=patterns or tuple(r.block_ids for r in solution.routes)!=state.routes:
                counts['canonical_changed'] += 1;return None
        except ValueError:
            counts['invalid'] += 1;return None
        signature = self.identity(solution,state.directions)
        if signature in self.evaluated:
            counts['duplicate'] += 1;return None
        after = self.state(state.routes,state.directions,blocks)
        old_cycles,new_cycles = self.cycles(current.state),self.cycles(after)
        candidate = BottleneckAction(solution,after.directions,after,operator,tuple(parents),signature,
                                     tuple(b-a for a,b in zip(old_cycles,new_cycles)),{})
        candidate.transfers = actual_parent_transfers(current,candidate,self.config,
            before_blocks=self.blocks(current.solution.patterns),after_blocks=blocks,parents=set(parents))
        counts['unique'] += 1
        counts['cross_robot_candidates'] += bool(candidate.transfers)
        counts['cross_parent_transfers'] += len(candidate.transfers)
        return candidate

    def neighbours(self, current, schedule, operator):
        critical,receivers,parents = self.bottleneck(current,schedule)
        selected = {p.parent_id:p for p in current.solution.patterns}
        blocks = self.blocks(current.solution.patterns)
        streams = []
        if operator=='two_opt':
            route,bits = current.state.routes[critical],current.directions[critical]
            pairs = [(i,j) for i in range(len(route)) for j in range(i+2,len(route)+1)]
            self.rng.shuffle(pairs)
            result = []
            for i,j in pairs:
                routes = list(current.state.routes);vectors = list(current.directions)
                routes[critical] = route[:i]+tuple(reversed(route[i:j]))+route[j:]
                vectors[critical] = bits[:i]+tuple(1-d for d in reversed(bits[i:j]))+bits[j:]
                affected = tuple(dict.fromkeys(blocks[b].parent_id for b in route[i:j]))
                state = self.state(routes,vectors,blocks)
                item = self.finish(current,current.solution.patterns,state,operator,affected)
                if item is not None:result.append(item)
                if len(result)>=self.candidate_limit:break
            return result
        for parent in parents[:4]:
            for receiver in receivers:
                left = self._transfer_assignments(current,parent,critical,receiver,selected[parent])
                if not left:continue
                partners = (None,)
                if operator=='swap':
                    partners = tuple(dict.fromkeys(blocks[b].parent_id for b in current.state.routes[receiver]
                        if blocks[b].parent_id!=parent))
                    partners = tuple(sorted(partners,key=lambda p:(-sum(self.config.process_time(b.length)
                        for b in blocks.values() if b.parent_id==p),p)))[:3]
                for partner in partners:
                    right = (None,) if partner is None else self._transfer_assignments(
                        current,partner,receiver,critical,selected[partner])
                    if not right:continue
                    removed = (parent,) if partner is None else (parent,partner)
                    partial = self.partial(current,set(removed))
                    options = []
                    for first in left:
                        assignment = self.space.actions[first]
                        changed = {**selected,parent:assignment.pattern}
                        for a,_,local_blocks in self.insert_parent(partial,assignment,blocks,positions=3,keep=6):
                            if partner is None:
                                options.append((tuple(changed.values()),a,removed));continue
                            for second in right:
                                other = self.space.actions[second]
                                both = {**changed,partner:other.pattern}
                                for b,_,_ in self.insert_parent(a,other,local_blocks,positions=3,keep=6):
                                    options.append((tuple(both.values()),b,removed))
                    self.rng.shuffle(options);streams.append(iter(options))
        # Round robin prevents a single receiving robot or parent pair filling
        # the proposal cap before another ownership structure is considered.
        result = []
        while streams and len(result)<self.candidate_limit:
            alive = []
            for stream in streams:
                try:patterns,state,removed = next(stream)
                except StopIteration:continue
                alive.append(stream)
                item = self.finish(current,patterns,state,operator,removed)
                if item is not None:result.append(item)
                if len(result)>=self.candidate_limit:break
            streams = alive
        return result

    def reconstruct(self, current, schedule, *, ordinal, ranker=None, width=4):
        critical,receivers,ordered = self.bottleneck(current,schedule)
        q = min(len(self.parents),3+ordinal%4)
        if q<3:return []
        blocks = self.blocks(current.solution.patterns)
        removed = list(ordered[:min(2,len(ordered))])
        receive_parents = list(dict.fromkeys(blocks[b].parent_id for r in receivers for b in current.state.routes[r]))
        self.rng.shuffle(receive_parents)
        for parent in (*receive_parents,*ordered,*(p.parent_id for p in self.parents)):
            if parent not in removed:removed.append(parent)
            if len(removed)>=q:break
        removed = tuple(removed[:q])
        selected = {p.parent_id:p for p in current.solution.patterns}
        retained = {k:b for k,b in blocks.items() if b.parent_id not in removed}
        beam = [(self.partial(current,set(removed)),selected,retained,0.)]
        for step,parent in enumerate(removed):
            proposals = []
            choices = {}
            for index in self.actions_by_parent[parent]:
                assignment = self.space.actions[index]
                key = assignment.pattern.kind,assignment.robots
                previous = choices.get(key)
                if previous is None or assignment.pattern==selected[parent]:choices[key] = index
            indices = list(choices.values());self.rng.shuffle(indices)
            # Rotate through legal mode/owner structures, without a Pmax band.
            indices = indices[:12]
            for state,patterns,local_blocks,total in beam:
                for index in indices:
                    assignment = self.space.actions[index]
                    for after,actions,after_blocks in self.insert_parent(state,assignment,local_blocks,positions=3,keep=4):
                        proposals.append(dict(state=after,patterns={**patterns,parent:assignment.pattern},
                            blocks=after_blocks,total=total,index=index,before=state,actions=actions,
                            pending=removed[step:]))
            if not proposals:return []
            scores = ranker(current,schedule,proposals) if ranker is not None else [self.ranking_rng.random() for _ in proposals]
            for proposal,score in zip(proposals,scores):proposal['total'] += float(score)
            order = sorted(range(len(proposals)),key=lambda i:-proposals[i]['total'])
            kept,structures = [],set()
            for i in order:
                proposal = proposals[i]
                structure = tuple((p.parent_id,p.pattern_id) for p in proposal['patterns'].values()),tuple(
                    (r,tuple(sorted(b for b in route if self._parent_id_from_blocks(b,proposal['blocks']) in removed)))
                    for r,route in enumerate(proposal['state'].routes))
                if structure in structures:continue
                structures.add(structure)
                kept.append((proposal['state'],proposal['patterns'],proposal['blocks'],proposal['total']))
                if len(kept)>=width:break
            beam = kept
        result = []
        for state,patterns,_,total in beam:
            item = self.finish(current,tuple(patterns.values()),state,'lns',removed)
            if item is not None:item.score = total/max(1,len(removed));result.append(item)
        return result

    @staticmethod
    def _parent_id_from_blocks(block_id, blocks):
        return blocks[block_id].parent_id


def run_bottleneck_vns(initial, workspace, native_evaluator, *, scorer=None, repair_ranker=None,
                       native_budget=500, max_states=80, max_shakes=8, batch_size=24, observer=None):
    """One VNS: native Cmax improvement, parent LNS escape, final B32 outside."""
    started = time.perf_counter()
    cache,archive = {},[]
    native_calls = 0
    best = None
    def evaluate(candidate, origin=None):
        nonlocal native_calls,best
        if candidate.signature in cache:return cache[candidate.signature]
        if native_calls>=native_budget:return None
        clock = time.perf_counter()
        schedule = native_evaluator(candidate.solution,workspace.config,candidate.directions)
        elapsed = time.perf_counter()-clock
        native_calls += 1;workspace.evaluated.add(candidate.signature)
        cache[candidate.signature] = schedule
        counts = workspace.counts[candidate.operator]
        counts['native_calls'] += 1;counts['native_feasible'] += int(schedule.feasible)
        counts['native_cross_robot_candidates'] += bool(candidate.transfers)
        counts['native_cross_parent_transfers'] += len(candidate.transfers)
        if schedule.feasible and (best is None or schedule.cmax<best[1].cmax):best = candidate,schedule
        if not schedule.feasible:archive.append((candidate,schedule))
        if observer is not None:observer(candidate,schedule,origin,elapsed)
        return schedule
    starts = [(candidate,evaluate(candidate)) for candidate in initial]
    starts = [(c,s) for c,s in starts if s is not None]
    if not starts:raise ValueError('BVNS has no complete initial candidate')
    current,schedule = best if best is not None else starts[0]
    initial_selected = current
    current_states = set()
    neighbourhood = shakes = states = 0
    operators = ('relocate','swap','two_opt')
    stop = 'state budget exhausted'
    while native_calls<native_budget and states<max_states:
        states += 1
        current_states.add(current.signature)
        operator = operators[neighbourhood]
        candidates = workspace.neighbours(current,schedule,operator)
        if schedule.feasible:
            kept = [c for c in candidates if max(workspace.cycles(c.state))<schedule.cmax]
            workspace.counts[operator]['lb_pruned'] += len(candidates)-len(kept)
            candidates = kept
        if scorer is not None and candidates:
            for c,score in zip(candidates,scorer(current,schedule,candidates)):c.score = float(score)
        else:
            for c in candidates:c.score = workspace.ranking_rng.random()
        candidates.sort(key=lambda c:-c.score)
        distinct,fill,structures = [],[],set()
        for candidate in candidates:
            structure = workspace.ownership_structure(candidate)
            if structure in structures:fill.append(candidate)
            else:structures.add(structure);distinct.append(candidate)
        prefix = distinct[:batch_size//2]
        prefix_ids = {id(c) for c in prefix}
        ordered = prefix+[c for c in candidates if id(c) not in prefix_ids]
        evaluated = []
        for candidate in ordered[:batch_size]:
            result = evaluate(candidate,current)
            if result is not None:evaluated.append((candidate,result))
        improving = [(c,s) for c,s in evaluated if s.feasible and
                     (not schedule.feasible or s.cmax<schedule.cmax)]
        if improving:
            current,schedule = min(improving,key=lambda x:x[1].cmax)
            workspace.counts[operator]['accepted'] += 1;neighbourhood = 0
            workspace.counts[operator]['accepted_parent_transfers'] += len(current.transfers)
            continue
        neighbourhood += 1
        if neighbourhood<len(operators):continue
        neighbourhood = 0
        if shakes>=max_shakes:stop = 'LNS perturbation budget exhausted';break
        proposals = workspace.reconstruct(current,schedule,ordinal=shakes,ranker=repair_ranker)
        shakes += 1
        if scorer is not None and proposals:
            for c,score in zip(proposals,scorer(current,schedule,proposals)):c.score = float(score)
        proposals.sort(key=lambda c:-c.score)
        tested = [(candidate,evaluate(candidate,current)) for candidate in proposals]
        tested = [(c,s) for c,s in tested if s is not None]
        if not tested:continue
        feasible = [(c,s) for c,s in tested if s.feasible]
        failed = [(c,s) for c,s in tested if not s.feasible]
        # Failed native scheduling is a search branch, not a proof. Periodic
        # GNN-ranked failed perturbations get local repair even if a feasible
        # incumbent exists; that incumbent remains in best.
        if failed and (not feasible or shakes%2==0):
            current,schedule = max(failed,key=lambda x:x[0].score)
            workspace.counts['lns']['native_deadlock_branches'] += 1
        else:current,schedule = min(feasible,key=lambda x:x[1].cmax)
        workspace.counts['lns']['accepted'] += 1
        workspace.counts['lns']['accepted_parent_transfers'] += len(current.transfers)
    if native_calls>=native_budget:stop = 'native evaluation budget exhausted'
    # Initial proposals have no action score. Their default zero must not
    # outrank every genuine GNN alternative when learned logits are negative.
    backups = [entry for entry in archive if entry[0].operator!='initial'] or archive
    selected = best if best is not None else max(backups,key=lambda x:x[0].score)
    return dict(candidate=selected[0],native=selected[1],initial_selected=initial_selected,native_calls=native_calls,
        states=len(current_states),rounds=states,shakes=shakes,
        stop=stop,counts={k:dict(v) for k,v in workspace.counts.items()},
        route_cache_hits=workspace.route_cache_hits,route_cache_builds=workspace.route_cache_builds,
        native_deadlock_reserve=len(archive),runtime_s=time.perf_counter()-started)


@dataclass(frozen=True)
class JointChildMove:
    block_id: str
    source: int
    target: int
    position: int
    direction: int


@dataclass
class ChildJointAction(BottleneckAction):
    moves: tuple = ()


class ChildJointWorkspace(BottleneckWorkspace):
    """Atomic child actions; inherited code supplies only route/geometry caches."""
    operators = ('initial','relocate_child','swap_children','cycle3','cycle4','mode')

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.counts = {name:Counter() for name in self.operators}
        self.comparison_cache={}

    def wrap(self, solution, directions, operator='initial'):
        bits=tuple(map(tuple,directions));routes=tuple(r.block_ids for r in solution.routes)
        state=self.state(routes,bits,self.blocks(solution.patterns))
        return ChildJointAction(solution,bits,state,operator,(),self.identity(solution,bits),(0.,)*4,{},moves=())

    def noop(self,current):
        return ChildJointAction(current.solution,current.directions,current.state,'noop',(),
            current.signature,(0.,)*4,{},moves=())

    def make(self,current,moves,operator,*,patterns=None):
        """Commit every removal before any insertion. No intermediate objective test."""
        self.counts[operator]['generated'] += 1
        moves=tuple(moves)
        if not moves or len({m.block_id for m in moves})!=len(moves):return None
        old_patterns={p.parent_id:p for p in current.solution.patterns}
        selected=old_patterns if patterns is None else {p.parent_id:p for p in patterns}
        changed={p for p in selected if selected[p]!=old_patterns[p]}
        if changed and operator!='mode':raise ValueError('ordinary child moves cannot change parent modes')
        old_blocks=self.blocks(current.solution.patterns)
        blocks=self.blocks(tuple(selected.values()))
        owner={b:r for r,route in enumerate(current.state.routes) for b in route}
        for move in moves:
            if move.block_id not in blocks or move.target not in range(4) or move.direction not in (0,1):return None
            block=blocks[move.block_id]
            if move.target not in joint_block_robots(block,selected[block.parent_id],self.config):return None
            if block.parent_id not in changed and owner.get(move.block_id)!=move.source:return None
        removed={m.block_id for m in moves}|{b for b,block in old_blocks.items() if block.parent_id in changed}
        routes=[];bits=[]
        for route,vector in zip(current.state.routes,current.directions):
            keep=[i for i,b in enumerate(route) if b not in removed]
            routes.append([route[i] for i in keep]);bits.append([vector[i] for i in keep])
        for move in sorted(moves,key=lambda m:(m.target,m.position,m.block_id)):
            r,k=move.target,move.position
            if not 0<=k<=len(routes[r]):return None
            routes[r].insert(k,move.block_id);bits[r].insert(k,move.direction)
        if sorted(b for row in routes for b in row)!=sorted(blocks):return None
        patterns=tuple(sorted(selected.values(),key=lambda p:p.parent_id))
        solution=CanonicalSolution(self.parents,patterns,tuple(Route(r,tuple(row)) for r,row in enumerate(routes)))
        state=self.state(routes,bits,blocks);signature=self.identity(solution,state.directions)
        if signature==current.signature:return None
        normalized=tuple(JointChildMove(m.block_id,m.source,m.target,routes[m.target].index(m.block_id),m.direction) for m in moves)
        parents=tuple(sorted({blocks[m.block_id].parent_id for m in moves}|changed))
        delta=tuple(b-a for a,b in zip(self.cycles(current.state),self.cycles(state)))
        return ChildJointAction(solution,state.directions,state,operator,parents,signature,delta,{},moves=normalized)

    def finish_child(self,current,preview):
        if preview is None:return None
        c=self.counts[preview.operator]
        if preview.signature in self.raw_seen or preview.signature in self.evaluated:
            c['duplicate']+=1;return None
        self.raw_seen.add(preview.signature)
        blocks=self.blocks(preview.solution.patterns)
        points={r:oriented_endpoints(blocks[row[0]],preview.directions[r][0])[0]
                for r,row in enumerate(preview.state.routes) if row}
        if _initial_error(points,self.config) is not None:c['invalid']+=1;return None
        try:
            canonical=canonicalize(self.parents,preview.solution.patterns,dict(enumerate(preview.state.routes)),
                                   self.config,scope=self.scope)
        except ValueError:c['invalid']+=1;return None
        if canonical.patterns!=preview.solution.patterns or tuple(r.block_ids for r in canonical.routes)!=preview.state.routes:
            c['canonical_changed']+=1;return None
        preview.solution=canonical
        preview.transfers=actual_parent_transfers(current,preview,self.config,
            before_blocks=self.blocks(current.solution.patterns),after_blocks=blocks,parents=set(preview.parents))
        c['unique']+=1;c['cross_robot_candidates']+=bool(preview.transfers)
        c['cross_parent_transfers']+=len(preview.transfers)
        c['cross_child_transfers']+=sum(m.source>=0 and m.source!=m.target for m in preview.moves)
        return preview

    def between(self,origin,target):
        """Describe a retained full plan in one reference state, including all affected routes."""
        if origin.signature==target.signature:return self.noop(origin)
        key=origin.signature,target.signature
        if key in self.comparison_cache:return self.comparison_cache[key]
        old_blocks=self.blocks(origin.solution.patterns);new_blocks=self.blocks(target.solution.patterns)
        old={b:(r,k,d) for r,(row,bits) in enumerate(zip(origin.state.routes,origin.directions))
             for k,(b,d) in enumerate(zip(row,bits))}
        old_patterns={p.parent_id:p for p in origin.solution.patterns}
        changed={p.parent_id for p in target.solution.patterns if old_patterns[p.parent_id]!=p}
        from difflib import SequenceMatcher
        stays=set()
        for r,(row,bits) in enumerate(zip(target.state.routes,target.directions)):
            stable={b for b,d in zip(row,bits) if b in old and old[b][0]==r and old[b][2]==d
                    and new_blocks[b].parent_id not in changed}
            before=[b for b in origin.state.routes[r] if b in stable]
            after=[b for b in row if b in stable]
            for match in SequenceMatcher(None,before,after,autojunk=False).get_matching_blocks():
                stays.update(before[match.a:match.a+match.size])
        moves=[]
        for r,(row,bits) in enumerate(zip(target.state.routes,target.directions)):
            for k,(b,d) in enumerate(zip(row,bits)):
                if b not in stays:
                    moves.append(JointChildMove(b,old.get(b,(-1,))[0],r,k,d))
        parents=tuple(sorted({new_blocks[m.block_id].parent_id for m in moves}|changed))
        delta=tuple(b-a for a,b in zip(self.cycles(origin.state),self.cycles(target.state)))
        transfers=actual_parent_transfers(origin,target,self.config,before_blocks=old_blocks,after_blocks=new_blocks)
        action=ChildJointAction(target.solution,target.directions,target.state,'mode' if changed else target.operator,parents,
            target.signature,delta,transfers,target.score,tuple(moves))
        self.comparison_cache[key]=action
        return action

    def ownership_structure(self,candidate):
        return tuple((p.pattern_id,tuple((b.block_id,r) for r,row in enumerate(candidate.state.routes)
            for b in self.space.children[p] if b.block_id in row)) for p in candidate.solution.patterns)

    def _joint_scores(self,current,schedule,candidates,ranker):
        if not candidates:return []
        values=ranker(current,schedule,candidates) if ranker is not None else [self.ranking_rng.random() for _ in candidates]
        for c,v in zip(candidates,values):
            c.score=float(v);self.counts[c.operator]['scored']+=1
        return candidates

    def _representative(self,current,pairs,operator,*,patterns=None):
        blocks=self.blocks(current.solution.patterns)
        removed={b for b,_ in pairs}
        target_patterns=current.solution.patterns if patterns is None else patterns
        new_blocks=self.blocks(target_patterns)
        changed={p.parent_id for p in target_patterns if p not in current.solution.patterns}
        removed|={b for b,x in blocks.items() if x.parent_id in changed}
        owner={b:(r,k,current.directions[r][k]) for r,row in enumerate(current.state.routes) for k,b in enumerate(row)}
        available=[sum(b not in removed for b in row) for row in current.state.routes]
        moves=[]
        for b,r in pairs:
            source,k,d=owner.get(b,(-1,0,0))
            # A neutral representative, not a MOVE/load-ranked insertion.
            position=self.rng.randrange(available[r]+1);available[r]+=1
            if source==r and len(pairs)==1:d=1-d
            moves.append(JointChildMove(b,source,r,position,d))
        return self.make(current,moves,operator,patterns=target_patterns)

    def proposals(self,current,schedule,ranker,*,limit=24,decode_width=2):
        """Learn child/robot relations first, then decode complete joint insertions."""
        blocks=self.blocks(current.solution.patterns);patterns={p.parent_id:p for p in current.solution.patterns}
        singles=[];by_pair={}
        for source,row in enumerate(current.state.routes):
            for b in row:
                for target in joint_block_robots(blocks[b],patterns[blocks[b].parent_id],self.config):
                    c=self._representative(current,((b,target),),'relocate_child')
                    if c is not None:singles.append(c)
        self._joint_scores(current,schedule,singles,ranker)
        for c in sorted(singles,key=lambda c:-c.score):
            m=c.moves[0]
            if m.source!=m.target:
                bucket=by_pair.setdefault((m.source,m.target),[])
                if len(bucket)<2:bucket.append(m.block_id)
        pools={'relocate_child':singles,'swap_children':[],'cycle3':[],'cycle4':[],'mode':[]}
        for q in (2,3,4):
            operator={2:'swap_children',3:'cycle3',4:'cycle4'}[q]
            for robots in itertools.permutations(range(4),q):
                if robots[0]!=min(robots):continue
                edges=tuple(zip(robots,robots[1:]+robots[:1]))
                choices=[by_pair.get(edge,()) for edge in edges]
                if not all(choices):continue
                for children in itertools.product(*choices):
                    c=self._representative(current,tuple((b,t) for b,(_,t) in zip(children,edges)),operator)
                    if c is not None:pools[operator].append(c)
        for assignment in self.space.actions:
            if assignment.pattern==patterns[assignment.parent_id]:continue
            selected=tuple(assignment.pattern if p.parent_id==assignment.parent_id else p for p in current.solution.patterns)
            pairs=tuple((b.block_id,r) for b,r in zip(self.space.children[assignment.pattern],assignment.robots))
            c=self._representative(current,pairs,'mode',patterns=selected)
            if c is not None:pools['mode'].append(c)
        templates=[]
        caps={'relocate_child':6,'swap_children':4,'cycle3':3,'cycle4':3,'mode':2}
        for operator,pool in pools.items():
            if operator!='relocate_child':self._joint_scores(current,schedule,pool,ranker)
            structures=set()
            for c in sorted(pool,key=lambda c:-c.score):
                structure=tuple((m.block_id,m.source,m.target) for m in c.moves),c.solution.patterns
                if structure in structures:continue
                structures.add(structure);templates.append(c)
                if len(structures)>=caps[operator]:break
        beams={i:[c] for i,c in enumerate(templates)}
        for step in range(max((len(c.moves) for c in templates),default=0)):
            expanded={};unique={}
            for i,beam in beams.items():
                rows=[]
                for c in beam:
                    if step>=len(c.moves):rows.append(c);continue
                    move=c.moves[step]
                    length=len(c.state.routes[move.target])-1
                    positions=bounded_insertion_positions(length,4)
                    for k in positions:
                        for d in (0,1):
                            moves=list(c.moves);moves[step]=replace(move,position=k,direction=d)
                            trial=self.make(current,moves,c.operator,patterns=c.solution.patterns)
                            if trial is not None:rows.append(trial);unique.setdefault(trial.signature,trial)
                expanded[i]=rows
            self._joint_scores(current,schedule,list(unique.values()),ranker)
            for i,rows in expanded.items():
                rows=list({c.signature:unique.get(c.signature,c) for c in rows}.values())
                beams[i]=sorted(rows,key=lambda c:-c.score)[:decode_width]
        previews=[c for rows in beams.values() for c in rows]
        candidates=[]
        for c in sorted(previews,key=lambda c:-c.score):
            complete=self.finish_child(current,c)
            if complete is not None:candidates.append(complete)
        # Keep whole ownership alternatives; all are still ranked by this GNN.
        diverse=[];taken=set()
        for operator in pools:
            for c in (c for c in candidates if c.operator==operator):
                if sum(x.operator==operator for x in diverse)>=2:break
                diverse.append(c);taken.add(c.signature)
        ordered=diverse+[c for c in candidates if c.signature not in taken]
        return ordered[:limit]


def run_child_joint(initial,workspace,native_evaluator,*,ranker=None,native_budget=500,
                    max_states=80,max_shakes=8,width=3,observer=None,decision_observer=None):
    """Learned relative B32 preference controls current, state bank and final plan.

    The no-GNN ablation uses the same child action space with native acceptance.
    No B32 is called here. Native DEADLOCK is never a hard candidate filter.
    """
    cache={};calls=0;states=0;shakes=0;expanded=set();accepted=set();visited=[]
    def evaluate(c,origin=None):
        nonlocal calls
        if c.signature in cache:return cache[c.signature]
        if calls>=native_budget:return None
        clock=time.perf_counter();s=native_evaluator(c.solution,workspace.config,c.directions)
        elapsed=time.perf_counter()-clock;calls+=1;cache[c.signature]=s;workspace.evaluated.add(c.signature)
        counts=workspace.counts[c.operator];counts['native_calls']+=1;counts['native_feasible']+=int(s.feasible)
        counts['native_cross_robot_candidates']+=bool(c.transfers)
        counts['native_cross_parent_transfers']+=len(c.transfers)
        if observer is not None:observer(c,s,origin,elapsed)
        return s
    starts=[c for c in initial if evaluate(c) is not None]
    if not starts:raise ValueError('child joint search has no complete GNN construction')
    best=starts[0];initial_selected=best;bank=list(starts)
    def preferences(reference,plans):
        actions=[workspace.between(reference,c) for c in plans]
        if ranker is not None:
            values=ranker(reference,cache[reference.signature],actions)
            return [float(v) for v in values]
        # Only the explicit no-GNN control uses native Cmax as its objective.
        return [(1,-cache[c.signature].cmax) if cache[c.signature].feasible else (0,0.) for c in plans]
    scores=preferences(best,[best,*starts[1:]])
    if len(starts)>1 and max(scores)>scores[0]:best=starts[max(range(len(scores)),key=lambda i:scores[i])]
    initial_selected=best;accepted.add(best.signature);current=best;stop='state check budget exhausted'
    while calls<native_budget and states<max_states:
        states+=1;expanded.add(current.signature);visited.append(current)
        proposed=workspace.proposals(current,cache[current.signature],ranker)
        tested=[c for c in proposed if evaluate(c,current) is not None]
        pool=list({c.signature:c for c in [best,*bank,*tested]}.values())
        reference=best;values=preferences(reference,pool);reference_value=values[0]
        eligible=[i for i,c in enumerate(pool) if c.signature not in accepted and values[i]>reference_value]
        improved=bool(eligible)
        if improved:
            index=max(eligible,key=lambda i:values[i]);best=workspace.between(reference,pool[index]);accepted.add(best.signature)
            workspace.counts[best.operator]['accepted']+=1
            workspace.counts[best.operator]['accepted_parent_transfers']+=len(best.transfers)
        if decision_observer is not None:decision_observer(reference,pool,values,best)
        order=sorted(range(len(pool)),key=lambda i:values[i],reverse=True);bank=[best];structures={workspace.ownership_structure(best)}
        for i in order:
            c=pool[i];structure=workspace.ownership_structure(c)
            if c.signature==best.signature or structure in structures:continue
            structures.add(structure);bank.append(c);workspace.counts[c.operator]['retained']+=1
            if len(bank)>=width:break
        if best.signature not in expanded:current=best;continue
        alternatives=[c for c in bank if c.signature not in expanded]
        if not alternatives:stop='no new learned ownership state';break
        if shakes>=max_shakes:stop='joint perturbation budget exhausted';break
        current=alternatives[0];shakes+=1;workspace.counts[current.operator]['perturbation']+=1
    if calls>=native_budget:stop='native evaluation budget exhausted'
    return dict(candidate=best,native=cache[best.signature],initial_selected=initial_selected,native_calls=calls,
        states=states,shakes=shakes,stop=stop,counts={k:dict(v) for k,v in workspace.counts.items()},
        visited_states=tuple(visited),
        runtime_objective='learned same-state B32 preference versus no-op' if ranker is not None else 'no-GNN native control',
        route_cache_hits=workspace.route_cache_hits,route_cache_builds=workspace.route_cache_builds)


class CandidateSourceKind(str, Enum):
    ATOMIC = "ATOMIC"
    LNS_REPAIRED = "LNS_REPAIRED"


class DestroyOperator(str, Enum):
    RANDOM_REMOVAL = "RANDOM_REMOVAL"
    CRITICAL_LOAD_REMOVAL = "CRITICAL_LOAD_REMOVAL"


class RepairOperator(str, Enum):
    GREEDY_REPAIR = "GREEDY_REPAIR"
    REGRET_2_REPAIR = "REGRET_2_REPAIR"


class BottleneckSelection(str, Enum):
    """Offline pilot selectors; deliberately outside production OPERATOR_PAIRS."""

    RANDOM = "RANDOM"
    LOAD = "LOAD"
    TRAVEL = "TRAVEL"
    WAIT_ANCHORED = "WAIT_ANCHORED"
    SHARED_Y_PAIR = "SHARED_Y_PAIR"


class CRRRBottleneck(str, Enum):
    WAIT_BOUNDARY = "WAIT_BOUNDARY"
    LOAD_BOUNDARY = "LOAD_BOUNDARY"


CRRR_OPERATOR = "CRRR"


OPERATOR_PAIRS = tuple(
    (destroy, repair)
    for destroy in DestroyOperator
    for repair in RepairOperator
)


@dataclass(frozen=True, order=True)
class CompleteCandidateIdentity:
    current_revision: int
    source_kind: str
    operator_identity: tuple[object, ...]


@dataclass(frozen=True)
class CompleteSearchCandidate:
    source_kind: CandidateSourceKind
    canonical_solution: CanonicalSolution
    candidate_identity: CompleteCandidateIdentity
    projected_process_makespan: float
    local_directed_travel_delta: float
    split_count_delta: int
    load_spread_delta: float
    provenance: tuple[tuple[str, object], ...]
    atomic_move: CandidateMove | None = None

    @property
    def cheap_score(self) -> tuple[object, ...]:
        return (
            self.projected_process_makespan,
            self.local_directed_travel_delta,
            self.split_count_delta,
            self.candidate_identity,
        )

    @property
    def solution(self) -> CanonicalSolution:
        return self.canonical_solution


@dataclass(frozen=True)
class PartialSearchState:
    parents: tuple[ParentWeld, ...]
    patterns: tuple[SplitPattern, ...]
    routes: tuple[tuple[str, ...], ...]
    removed_parent_ids: tuple[str, ...]
    source_revision: int


@dataclass(frozen=True)
class RepairResult:
    candidate: CompleteSearchCandidate | None
    insertion_evaluations: int
    rejection_reason: str | None = None
    directions: DirectionVectors | None = None
    actions: tuple[OrientedInsertionAction, ...] = ()
    direction_candidates: tuple[DirectionVectors, ...] = ()
    oriented_state: OrientedRouteState | None = None


@dataclass(frozen=True)
class OrientedInsertionAction:
    block_id: str
    robot: int
    position: int
    direction: int
    neighbour_changes: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class OrientedRouteState:
    routes: tuple[tuple[str, ...], ...]
    directions: tuple[tuple[int, ...], ...]
    loads: tuple[float, ...]
    arc_costs: tuple[tuple[float, ...], ...]

    @property
    def travel(self) -> float:
        return sum(sum(arcs) for arcs in self.arc_costs)


@dataclass(frozen=True)
class ChildTransfer:
    move: CandidateMove
    action: OrientedInsertionAction
    partial: OrientedRouteState
    state: OrientedRouteState
    solution: CanonicalSolution


@dataclass(frozen=True)
class CRRRRegion:
    bottleneck_type: CRRRBottleneck
    affected_parent_ids: tuple[str, ...]
    critical_robot: int
    partner_robot: int
    anchor_x: float
    wait_duration: float


@dataclass(frozen=True)
class _InsertionAlternative:
    parent_id: str
    routes: tuple[tuple[str, ...], ...]
    hints: tuple[tuple[int, ...], ...]
    score: tuple[object, ...]
    trace: tuple[tuple[str, int, int, int], ...]
    actions: tuple[OrientedInsertionAction, ...] = ()
    oriented_state: OrientedRouteState | None = None


def _parent_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def _optional_split_count(solution: CanonicalSolution, config: ScientificConfig) -> int:
    from mrta_reference.geometry import whole_eligible_rails

    parents = {parent.parent_id: parent for parent in solution.parents}
    return sum(
        pattern.kind is not SplitKind.WHOLE
        and bool(
            whole_eligible_rails(
                parents[pattern.parent_id].start,
                parents[pattern.parent_id].end,
                config,
            )
        )
        for pattern in solution.patterns
    )


def _directed_proxy(
    routes: Sequence[Sequence[str]],
    hints: Sequence[Sequence[int]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    total = 0.0
    for robot, route in enumerate(routes):
        for index in range(len(route) - 1):
            _, end = oriented_endpoints(blocks[route[index]], hints[robot][index])
            start, _ = oriented_endpoints(
                blocks[route[index + 1]], hints[robot][index + 1]
            )
            total += math.dist(end, start) / config.empty_speed
    return total


def _loads(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, float, float, float]:
    return tuple(
        sum(config.process_time(blocks[item].length) for item in route)
        for route in routes
    )  # type: ignore[return-value]


def _route_arc_costs(
    route: Sequence[str],
    hints: Sequence[int],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, ...]:
    """Cache the directed costs of one route without changing proxy semantics."""
    costs = []
    for index in range(len(route) - 1):
        _, end = oriented_endpoints(blocks[route[index]], hints[index])
        start, _ = oriented_endpoints(blocks[route[index + 1]], hints[index + 1])
        costs.append(math.dist(end, start) / config.empty_speed)
    return tuple(costs)


def _proxy_from_arc_costs(costs: Sequence[Sequence[float]]) -> float:
    # Keep the exact robot/arc accumulation order used by _directed_proxy.  A
    # sum-of-route-totals would be faster but can perturb floating-point ties.
    total = 0.0
    for route_costs in costs:
        for value in route_costs:
            total += value
    return total


def _insert_route_arc_costs(
    route: Sequence[str],
    hints: Sequence[int],
    costs: Sequence[float],
    block: WeldingBlock,
    position: int,
    hint: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, ...]:
    """Update only the one or two arcs touched by an insertion."""
    if not route:
        return ()
    before = tuple(costs[: max(0, position - 1)])
    after = tuple(costs[position:])
    inserted: list[float] = []
    if position:
        _, end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        start, _ = oriented_endpoints(block, hint)
        inserted.append(math.dist(end, start) / config.empty_speed)
    if position < len(route):
        _, end = oriented_endpoints(block, hint)
        start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        inserted.append(math.dist(end, start) / config.empty_speed)
    return before + tuple(inserted) + after


def _insert_tuple(values: tuple[object, ...], position: int, value: object) -> tuple[object, ...]:
    return values[:position] + (value,) + values[position:]


def _hints_from_current(
    solution: CanonicalSolution, directions: DirectionVectors
) -> dict[str, int]:
    return {
        block_id: directions[robot][position]
        for robot, route in enumerate(solution.routes)
        for position, block_id in enumerate(route.block_ids)
        if position < len(directions[robot])
    }


def atomic_complete_candidate(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    screened,
    config: ScientificConfig,
) -> CompleteSearchCandidate:
    current_blocks = block_map(current, config)
    provisional_blocks = block_map(screened.solution, config)
    current_routes = tuple(route.block_ids for route in current.routes)
    provisional_routes = tuple(route.block_ids for route in screened.solution.routes)
    hints_by_block = _hints_from_current(current, current_directions)
    current_hints = tuple(
        tuple(hints_by_block.get(item, 0) for item in route) for route in current_routes
    )
    provisional_hints = tuple(
        tuple(hints_by_block.get(item, 0) for item in route)
        for route in provisional_routes
    )
    current_loads = _loads(current_routes, current_blocks, config)
    new_loads = _loads(provisional_routes, provisional_blocks, config)
    from .neighborhood import candidate_key_order

    identity = CompleteCandidateIdentity(
        current.revision,
        CandidateSourceKind.ATOMIC.value,
        candidate_key_order(screened.candidate.key),
    )
    return CompleteSearchCandidate(
        CandidateSourceKind.ATOMIC,
        screened.solution,
        identity,
        max(new_loads),
        _directed_proxy(provisional_routes, provisional_hints, provisional_blocks, config)
        - _directed_proxy(current_routes, current_hints, current_blocks, config),
        _optional_split_count(screened.solution, config)
        - _optional_split_count(current, config),
        (max(new_loads) - min(new_loads))
        - (max(current_loads) - min(current_loads)),
        (("move_type", screened.candidate.key.move_type.value),),
        screened.candidate,
    )


def destroy_size(
    parent_count: int, *, rho: float, q_min: int, q_max: int
) -> int:
    if parent_count <= 0:
        return 0
    if rho < 0.0 or q_min < 1 or q_max < q_min:
        raise ValueError("invalid destroy-size configuration")
    q = max(q_min, min(q_max, round(rho * parent_count)))
    if parent_count > 1:
        return min(q, parent_count - 1)
    return 1


def _critical_parent_order(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
) -> tuple[str, ...]:
    blocks = block_map(current, config)
    parent_robots: dict[str, set[int]] = {
        parent.parent_id: set() for parent in current.parents
    }
    for robot, route in enumerate(current.routes):
        for block_id in route.block_ids:
            parent_robots[_parent_id(block_id)].add(robot)
    loads = _loads(tuple(route.block_ids for route in current.routes), blocks, config)
    waits = [0.0, 0.0, 0.0, 0.0]
    for operation in schedule.operations:
        if operation.kind is OperationKind.WAIT:
            waits[operation.robot_id] += operation.duration
    completions = schedule.robot_completion
    critical_robot = max(range(4), key=lambda robot: (completions[robot], -robot))

    def score(parent_id: str) -> tuple[object, ...]:
        robots = parent_robots[parent_id]
        return (
            -int(critical_robot in robots),
            -max((completions[robot] for robot in robots), default=0.0),
            -max((loads[robot] for robot in robots), default=0.0),
            -max((waits[robot] for robot in robots), default=0.0),
            parent_id,
        )

    return tuple(sorted(parent_robots, key=score))


def destroy_parents(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    operator: DestroyOperator,
    *,
    q: int,
    seed: int,
    config: ScientificConfig,
) -> PartialSearchState:
    parent_ids = tuple(parent.parent_id for parent in current.parents)
    if not (1 <= q <= len(parent_ids)):
        raise ValueError("q must select one or more existing parents")
    if operator is DestroyOperator.RANDOM_REMOVAL:
        removed = tuple(sorted(random.Random(seed).sample(parent_ids, q)))
    else:
        removed = tuple(sorted(_critical_parent_order(current, schedule, config)[:q]))
    return _remove_parent_set(current, removed)


def _remove_parent_set(
    current: CanonicalSolution, removed: Sequence[str]
) -> PartialSearchState:
    removed = tuple(sorted(set(removed)))
    removed_set = set(removed)
    routes = tuple(
        tuple(item for item in route.block_ids if _parent_id(item) not in removed_set)
        for route in current.routes
    )
    return PartialSearchState(
        current.parents,
        current.patterns,
        routes,
        removed,
        current.revision,
    )


def select_deadlock_blocker_parents(current, schedule, config, *, neighbour_layers=1,
                                    max_parents=None, seed=0):
    """Close direct terminal blocker tasks and one (at most two) route layers.

    Uses structured DEADLOCK evidence, including both MOVE endpoints. No WAIT
    durations, text diagnostics, geometric proxy, or random route removal.
    """
    from mrta_reference.model import ScheduleStatus
    if schedule.status is not ScheduleStatus.DEADLOCK:
        raise ValueError('deadlock parent selection requires DEADLOCK')
    if neighbour_layers not in (1, 2):
        raise ValueError('deadlock destruction supports one or two direct neighbour layers')
    blocks = block_map(current, config)
    known = {p.parent_id for p in current.parents}
    evidence = schedule.deadlock_blockages
    if max_parents is not None:
        if max_parents<4:
            raise ValueError('bounded deadlock selection needs room for both MOVE endpoints')
        pairs=[]
        for event in evidence:
            blocked=set(event.blocked_parent_ids)&known
            for operation,parents in zip(event.blocker_operations,event.blocker_parent_ids):
                direct=set(parents)&known
                if operation is not None and blocked and direct:
                    pairs.append((event,operation,blocked|direct))
        if not pairs:
            return None
        pairs.sort(key=lambda item:(-item[0].ready_time,item[0].robot_id,item[1].operation_id))
        event,operation,removed=pairs[seed%len(pairs)]
        removed=set(removed)
        # Keep atomic blocked/blocker pairs, including source/destination of
        # MOVE. Focus one actual blocked robot rather than union every robot's
        # entire committed prefix. No cycle test is required.
        for other,_,core in pairs:
            if other.robot_id==event.robot_id and len(removed|core)<=max_parents-2:
                removed.update(core)
        neighbours=[]
        for route in current.routes:
            for position,block in enumerate(route.block_ids):
                if blocks[block].parent_id in removed:
                    for index in (position-1,position+1):
                        if 0<=index<len(route.block_ids):
                            neighbours.append(blocks[route.block_ids[index]].parent_id)
        for parent in dict.fromkeys(neighbours):
            if len(removed)>=max_parents:
                break
            removed.add(parent)
        return _remove_parent_set(current,removed)
    blocking = {parent for event in evidence for parents in event.blocker_parent_ids
                for parent in parents if parent in known}
    if not blocking:
        return None
    removed = blocking | {parent for event in evidence for parent in event.blocked_parent_ids if parent in known}
    for _ in range(neighbour_layers):
        adjacent = set(removed)
        for route in current.routes:
            for position, block in enumerate(route.block_ids):
                if blocks[block].parent_id not in removed:
                    continue
                for index in (position-1, position+1):
                    if 0 <= index < len(route.block_ids):
                        adjacent.add(blocks[route.block_ids[index]].parent_id)
        removed = adjacent
    return _remove_parent_set(current, removed)


def schedule_feedback_summary(current, schedule, blockers, config):
    """Observed timeline costs and exact selected-rollout WAIT evidence only.

    Forbidden interval coverage describes observed blockers, not a causal WAIT
    decomposition. Geometry/proximity never creates a blocking relationship.
    """
    if not schedule.feasible:
        raise ValueError('schedule feedback requires a feasible incumbent')
    blocks = block_map(current, config)
    processing, move, robot_wait = [0.] * 4, [0.] * 4, [0.] * 4
    parent_costs = {p.parent_id: [0.] * 4 for p in current.parents}
    parent_wait, parent_blocked, parent_move = {}, {}, {}
    parent_processing={p.parent_id:[0.]*4 for p in current.parents}
    critical=max(range(4),key=lambda r:(schedule.robot_completion[r],-r))
    parent_critical_wait,parent_critical_blocked={},{}
    block_processing, block_move, block_wait = (dict.fromkeys(blocks, 0.) for _ in range(3))
    mapped_wait = {event.waiting.operation_id: event.delayed_block_id for event in blockers}
    operation_by_id={op.operation_id:op for op in schedule.operations}

    def arc_source(block_id,robot):
        route=current.routes[robot].block_ids
        if block_id in route:
            position=route.index(block_id)
            return route[position-1] if position else None
        return None

    def block_for(op):
        block_id = mapped_wait.get(op.operation_id, op.block_id)
        if block_id in blocks:
            return block_id
        # The scheduler's MOVE/WAIT operation ID retains its destination index.
        if not op.operation_id.startswith(f'R{op.robot_id}:B'):
            return None
        try:
            index = int(op.operation_id.split(':B', 1)[1].split(':', 1)[0])
            if index < 0:
                return None
            return current.routes[op.robot_id].block_ids[index]
        except (IndexError, KeyError, ValueError):
            return None

    for op in schedule.operations:
        block_id = block_for(op)
        parent = blocks[block_id].parent_id if block_id in blocks else None
        if op.kind is OperationKind.WAIT:
            robot_wait[op.robot_id] += op.duration
            if parent is not None:
                parent_wait[parent] = parent_wait.get(parent, 0.) + op.duration
                block_wait[block_id] += op.duration
                if op.robot_id==critical:
                    parent_critical_wait[parent]=parent_critical_wait.get(parent,0.)+op.duration
        elif op.kind is OperationKind.MOVE:
            move[op.robot_id] += op.duration
            if parent is not None:
                parent_move[parent] = parent_move.get(parent, 0.) + op.duration
                block_move[block_id] += op.duration
        else:
            processing[op.robot_id] += op.duration
            if parent is not None:
                block_processing[block_id] += op.duration
                parent_processing[parent][op.robot_id]+=op.duration
        if parent is not None:
            parent_costs[parent][op.robot_id] += op.duration
    for event in blockers:
        by_parent = {}
        for operation_id, block_id, intervals in event.blocker_intervals:
            if block_id in blocks:
                by_parent.setdefault(blocks[block_id].parent_id, []).extend(intervals)
            operation=operation_by_id.get(operation_id)
            if event.waiting.robot_id==critical and operation is not None and operation.kind is OperationKind.MOVE:
                source=arc_source(block_id,operation.robot_id)
                if source in blocks:
                    parent_critical_blocked[blocks[source].parent_id]=max(
                        parent_critical_blocked.get(blocks[source].parent_id,0.),sum(hi-lo for lo,hi in intervals))
        if event.waiting.robot_id==critical and event.delayed_operation.kind is OperationKind.MOVE:
            source=arc_source(event.delayed_block_id,critical)
            if source in blocks:
                parent_critical_wait[blocks[source].parent_id]=max(
                    parent_critical_wait.get(blocks[source].parent_id,0.),event.waiting.duration)
        for parent, intervals in by_parent.items():
            end, covered = -math.inf, 0.
            for lo, hi in sorted(intervals):
                covered += max(0., hi - max(lo, end))
                end = max(end, hi)
            parent_blocked[parent] = parent_blocked.get(parent, 0.) + covered
            if event.waiting.robot_id==critical:
                parent_critical_blocked[parent]=parent_critical_blocked.get(parent,0.)+covered
    maximum = max(max(schedule.robot_completion), config.numeric_epsilon)
    peak_load = max(max(processing), config.numeric_epsilon)
    criticality = [schedule.robot_completion[r] / maximum * (.5 + .5 * processing[r] / peak_load)
                   for r in range(4)]
    scores = {parent: sum(cost * criticality[r] for r, cost in enumerate(costs))
                       + parent_blocked.get(parent, 0.) for parent, costs in parent_costs.items()}
    return dict(completion=schedule.robot_completion, robot_wait=tuple(robot_wait),critical_robot=critical,
                parent_processing={p:tuple(costs) for p,costs in parent_processing.items()},
                parent_critical_wait=parent_critical_wait,parent_critical_blocked=parent_critical_blocked,
                parent_wait=parent_wait, parent_blocked=parent_blocked, scores=scores,
                processing=tuple(processing), move=tuple(move), parent_move=parent_move,
                parent_costs={parent: tuple(costs) for parent, costs in parent_costs.items()},
                block_processing=block_processing, block_move=block_move, block_wait=block_wait)


def select_routeaware_bottleneck(current, directions, schedule, blockers, config, *, q, seed,
                                 assignment_space=None, load_first=False):
    """Use the real critical/WAIT parents plus a costly directed route arc.

    Historical cached schedules without operations have no measured completion
    or WAIT. Their fallback selects actual MOVE arc endpoints and neighbours;
    it never invents a critical robot from ScheduleResult's zero defaults.
    Both callers use the same q/seed stream and whole-parent removal.
    """
    if not schedule.feasible:
        raise ValueError('route feedback requires a feasible schedule')
    if not 1 <= q <= len(current.parents):
        raise ValueError('invalid parent removal count')
    blocks = block_map(current, config)
    state = oriented_route_state(tuple(r.block_ids for r in current.routes), directions, blocks, config)
    arcs = sorted(((cost,r,i) for r,values in enumerate(state.arc_costs) for i,cost in enumerate(values)),
                  key=lambda item: (-item[0],item[1],item[2]))
    anchors=[];feedback=None
    if schedule.operations:
        feedback=schedule_feedback_summary(current,schedule,blockers,config)
        critical=max(range(4),key=lambda r:(schedule.robot_completion[r],-r))
        critical_parents={_parent_id(b) for b in current.routes[critical].block_ids}
        if load_first:
            contribution=feedback['parent_processing']
            ranked=sorted(critical_parents,key=lambda p:(-contribution[p][critical],p))
            anchors.append(ranked[seed%min(3,len(ranked))])
            peak=max(range(4),key=lambda r:(feedback['processing'][r],-r))
            peak_parents={_parent_id(b) for b in current.routes[peak].block_ids}
            ranked_peak=sorted(peak_parents,key=lambda p:(-contribution[p][peak],p))
            anchors.append(ranked_peak[(seed//3)%min(3,len(ranked_peak))])
            important=[arc for arc in arcs if arc[1] in (critical,peak)] or arcs
            if important:
                _,robot,index=important[seed%min(3,len(important))]
                route=state.routes[robot]
                anchors.extend(_parent_id(route[i]) for i in (index,index+1))
            waits=feedback['parent_critical_wait']
            blocking=feedback['parent_critical_blocked']
            if waits:
                anchors.append(min(waits,key=lambda p:(-waits[p],p)))
            if blocking:
                anchors.append(min(blocking,key=lambda p:(-blocking[p],p)))
            fallback=select_feedback_bottleneck(current,feedback,q=q,seed=seed,
                                                assignment_space=assignment_space,load_first=True)
            selected=list(dict.fromkeys((*anchors,*fallback.removed_parent_ids)))[:q]
            if len(selected)<q:
                selected.extend(p for p in sorted(contribution) if p not in selected)
                selected=selected[:q]
            return _remove_parent_set(current,selected),dict(source='LoadFirst B32 P/M/critical WAIT',
                critical_robot=critical,processing_peak_robot=peak,
                anchors=list(dict.fromkeys(anchors)),feedback=feedback)
        anchors.append(min(critical_parents,key=lambda p:(-feedback['parent_costs'][p][critical],p)))
        waits=sorted((p for p,w in feedback['parent_wait'].items() if w>config.numeric_epsilon),
                     key=lambda p:(-feedback['parent_wait'][p],p))
        if waits:
            anchors.append(waits[0])
        # Exact selected-rollout blocking intervals, not geometric proximity.
        blocked=sorted(feedback['parent_blocked'],key=lambda p:(-feedback['parent_blocked'][p],p))
        anchors.extend(blocked[:1])
        fallback=select_feedback_bottleneck(current,feedback,q=q,seed=seed,assignment_space=assignment_space)
        fill=list(fallback.removed_parent_ids)
    else:
        critical=None
        fill=sorted(p.parent_id for p in current.parents)
        random.Random(seed).shuffle(fill)
    if arcs:
        _,robot,index=arcs[(seed % min(3,len(arcs)))]
        route=state.routes[robot]
        anchors.extend(_parent_id(route[i]) for i in (index,index+1))
        neighbours=[_parent_id(route[i]) for i in (index-1,index+2) if 0<=i<len(route)]
    else:
        neighbours=[]
    selected=list(dict.fromkeys((*anchors,*neighbours,*fill)))[:q]
    if len(selected)<q:
        selected.extend(p for p in sorted(b.parent_id for b in current.parents) if p not in selected)
        selected=selected[:q]
    return _remove_parent_set(current,selected), dict(
        source='B32 critical/WAIT + directed MOVE' if feedback is not None else
               'historical timeline missing; directed MOVE only',
        critical_robot=critical,anchors=list(dict.fromkeys(anchors)),
        feedback=feedback)


def select_feedback_bottleneck(current, feedback, *, q, seed, assignment_space=None, load_first=False):
    """Remove about 70% from the real critical robot, closing whole parents.

    Legal mode/owner alternatives take priority on the critical robot. Other
    routes contribute high MOVE or exchangeable parents. Random weighted ranks
    vary the subproblem reproducibly using actual timeline costs, never geometry
    as a substitute for WAIT.
    """
    if q == 0 and len(current.parents) == 1:
        return _remove_parent_set(current, ())
    if not 1 <= q <= len(current.parents):
        raise ValueError('invalid parent removal count')
    scores = feedback['scores']
    rng = random.Random(seed)
    critical = max(range(4), key=lambda r: (feedback['completion'][r], -r))
    owners = {block: r for r, route in enumerate(current.routes) for block in route.block_ids}
    critical_parents = {_parent_id(b) for b in current.routes[critical].block_ids}
    flexible, exchangeable = set(), set()
    if assignment_space is not None:
        patterns = {p.parent_id: p for p in current.patterns}
        assigned = {p: tuple(owners[b.block_id] for b in assignment_space.children[pattern])
                    for p, pattern in patterns.items()}
        for action in assignment_space.actions:
            if action.pattern != patterns[action.parent_id] or action.robots != assigned[action.parent_id]:
                flexible.add(action.parent_id)
            if action.robots != assigned[action.parent_id]:
                exchangeable.add(action.parent_id)
    costs = feedback['parent_processing'] if load_first else feedback['parent_costs']
    if load_first:
        scores={p:costs[p][critical] for p in costs}
    moves = feedback['parent_move']

    def rank(pool, *, on_critical):
        ranked = []
        for parent in sorted(pool):
            weight = (costs[parent][critical] + scores[parent] if on_critical
                      else scores[parent] + moves.get(parent, 0.))
            if load_first:
                weight=costs[parent][critical] if on_critical else sum(costs[parent])
            priority = parent in flexible if on_critical else parent in exchangeable
            ranked.append((not priority, -math.log(max(rng.random(), 1e-15)) / max(weight, 1e-9), parent))
        return [parent for _, _, parent in sorted(ranked)]

    critical_order = rank(critical_parents, on_critical=True)
    if critical_order:
        anchor = min(critical_parents, key=lambda p: (
            p not in flexible, -costs[p][critical], -scores[p], p))
        critical_order.remove(anchor)
        critical_order.insert(0, anchor)
    critical_quota = max(1, round(.7 * q))
    removed = critical_order[:critical_quota]
    others = rank(set(scores) - critical_parents, on_critical=False)
    removed.extend(others[:q - critical_quota])
    # Either route group can be too small. Fill from remaining contributors.
    remaining = sorted(set(scores) - set(removed), key=lambda p: (-scores[p], p))
    removed.extend(remaining[:q - len(removed)])
    return _remove_parent_set(current, removed)


def shared_y_parent_links(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
) -> tuple[tuple[str, str, float], ...]:
    """Observed cross-rail proximity, not causal blockers or collision evidence.

    Clip SETUP/WELD/POST to the shared Y band. Connect distinct parents when
    their swept boxes are within the existing interference distances, even
    when their visits are separated in time. Max aggregation avoids counting
    a parent repeatedly merely because its execution has more phases/children.
    """
    if not schedule.feasible:
        raise ValueError("shared-Y links require a feasible current schedule")
    parents = {p.parent_id: p for p in current.parents}
    visits = ([], [])
    waits = {p: 0.0 for p in parents}
    low, high = config.by
    eps = config.numeric_epsilon
    for op in schedule.operations:
        if op.block_id is None or _parent_id(op.block_id) not in parents:
            continue
        parent = _parent_id(op.block_id)
        if op.kind is OperationKind.WAIT:
            waits[parent] += op.duration
        if op.kind not in (OperationKind.SETUP, OperationKind.WELD, OperationKind.POST):
            continue
        if op.duration <= eps:
            continue
        dy = op.end[1] - op.start[1]
        if abs(dy) <= eps:
            if not low - eps <= op.start[1] <= high + eps:
                continue
            a, b = 0.0, 1.0
        else:
            ends = sorted(((low - op.start[1]) / dy, (high - op.start[1]) / dy))
            a, b = max(0.0, ends[0]), min(1.0, ends[1])
            if b <= a + eps:
                continue
        t0, t1 = op.start_time + a * op.duration, op.start_time + b * op.duration
        x0, y0 = op.point_at(t0)
        x1, y1 = op.point_at(t1)
        visits[op.robot_id // 2].append((parent, op.robot_id, t0, t1,
                                       min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    links: dict[tuple[str, str], float] = {}
    cmax = max(max(schedule.robot_completion), eps)
    for a in visits[0]:
        for b in visits[1]:
            if a[0] == b[0]:
                continue
            dx = max(0.0, a[4] - b[5], b[4] - a[5])
            dy = max(0.0, a[6] - b[7], b[6] - a[7])
            if dx > config.interference_dx + eps or dy > config.interference_dy + eps:
                continue
            gap = max(0.0, a[2] - b[3], b[2] - a[3])
            scale = max(a[3] - a[2], b[3] - b[2], config.t_pre + config.t_post, eps)
            criticality = max(schedule.robot_completion[a[1]], schedule.robot_completion[b[1]]) / cmax
            pressure = (waits[a[0]] + waits[b[0]]) / max(
                config.process_time(parents[a[0]].length) + config.process_time(parents[b[0]].length), eps)
            score = (0.25 + criticality) ** 2 * (1.0 + pressure) / (
                (1.0 + dx / config.interference_dx + dy / config.interference_dy) * (1.0 + gap / scale))
            pair = tuple(sorted((a[0], b[0])))
            links[pair] = max(links.get(pair, 0.0), score)
    return tuple((a, b, score) for (a, b), score in sorted(links.items()))


def select_bottleneck_parents(
    current: CanonicalSolution,
    directions: DirectionVectors,
    schedule: ScheduleResult,
    selection: BottleneckSelection,
    *,
    q: int,
    seed: int,
    config: ScientificConfig,
) -> PartialSearchState | None:
    """Select closed parent groups using only the current certified timeline.

    WAIT_ANCHORED uses observed waits and temporal/spatial neighbours, not
    inferred causal blockers. SHARED_Y_PAIR keeps both ends of an observed
    cross-rail proximity link (q >= 2). None means no positive WAIT or no
    shared-Y link, respectively; callers must explicitly handle that case.
    Directions are required so TRAVEL uses actual exit/entry endpoints.
    """
    if not schedule.feasible:
        raise ValueError("bottleneck selection requires a feasible current schedule")
    if not 1 <= q <= len(current.parents):
        raise ValueError("q must select one or more existing parents")
    if selection is BottleneckSelection.RANDOM:
        return destroy_parents(current, schedule, DestroyOperator.RANDOM_REMOVAL,
                               q=q, seed=seed, config=config)
    rng = random.Random(seed)
    blocks = block_map(current, config)
    patterns = {p.parent_id: p for p in current.patterns}
    owners = {item: r for r, route in enumerate(current.routes) for item in route.block_ids}
    parent_blocks = {p.parent_id: [] for p in current.parents}
    for block in blocks.values():
        parent_blocks[block.parent_id].append(block)
    legal = {}
    for item, block in blocks.items():
        pattern = patterns[block.parent_id]
        if pattern.kind is SplitKind.X_SPLIT:
            base = 0 if pattern.rail.value == "UPPER" else 2
            legal[item] = {base + int(item.endswith("::1"))}
        else:
            legal[item] = {r for r in range(4) if robot_is_eligible(block, r, config)}
    movable = {p: any(legal[b.block_id] - {owners[b.block_id]} for b in bs)
               for p, bs in parent_blocks.items()}
    loads = _loads(tuple(r.block_ids for r in current.routes), blocks, config)
    completion = schedule.robot_completion
    max_completion = max(max(completion), config.numeric_epsilon)

    def choose(rows):
        values, weights = zip(*rows)
        return rng.choices(values, weights=weights, k=1)[0]

    anchors: list[str] = []
    wait = None
    if selection is BottleneckSelection.SHARED_Y_PAIR:
        links = shared_y_parent_links(current, schedule, config)
        if not links:
            return None
        anchors = list(choose([((a, b), weight) for a, b, weight in links]))
        # For q=1, prefer the parent owned by a later-finishing robot.
        anchors.sort(key=lambda p: (-max(completion[owners[b.block_id]] for b in parent_blocks[p]), p))
    elif selection is BottleneckSelection.LOAD:
        # Both completion and processing matter. All legal robot moves count,
        # including left/right moves on the same rail.
        robot = choose([(r, max(loads[r], 1e-9) * (completion[r] / max_completion) ** 2)
                        for r in range(4) if current.routes[r].block_ids])
        candidates = sorted({_parent_id(b) for b in current.routes[robot].block_ids})
        flexible = [p for p in candidates if movable[p]]
        anchors = [choose([(p, sum(config.process_time(b.length)
                                  for b in parent_blocks[p] if owners[b.block_id] == robot))
                           for p in (flexible or candidates)])]
    elif selection is BottleneckSelection.TRAVEL:
        arcs = []
        for r, route in enumerate(current.routes):
            for i, cost in enumerate(_route_arc_costs(route.block_ids, directions[r], blocks, config)):
                arcs.append(((route.block_ids[i], route.block_ids[i + 1]),
                             max(cost, 1e-9) * (0.25 + completion[r] / max_completion)))
        if arcs:
            anchors = list(dict.fromkeys(_parent_id(b) for b in choose(arcs)))
        else:
            anchors = [rng.choice(current.parents).parent_id]
    elif selection is BottleneckSelection.WAIT_ANCHORED:
        waits = [op for op in schedule.operations
                 if op.kind is OperationKind.WAIT and op.duration > config.numeric_epsilon]
        if not waits:
            return None
        wait = choose([(op, op.duration * (0.25 + completion[op.robot_id] / max_completion) ** 2)
                       for op in waits])
        own = [op for op in schedule.operations if op.robot_id == wait.robot_id
               and op.block_id in blocks and op.kind is not OperationKind.WAIT]
        before = [op for op in own if op.end_time <= wait.start_time + config.numeric_epsilon]
        after = [op for op in own if op.start_time >= wait.end_time - config.numeric_epsilon]
        for op in ([max(before, key=lambda op: op.end_time)] if before else []) + (
            [min(after, key=lambda op: op.start_time)] if after else []
        ):
            if _parent_id(op.block_id) not in anchors:
                anchors.append(_parent_id(op.block_id))
        if not anchors and wait.block_id in blocks:
            anchors.append(_parent_id(wait.block_id))
    else:
        raise ValueError(f"unknown bottleneck selection: {selection}")

    anchor_blocks = [b for p in anchors for b in parent_blocks[p]]
    receiver_robots = set().union(*(legal[b.block_id] for b in anchor_blocks))
    adjacent = set()
    for route in current.routes:
        for i, item in enumerate(route.block_ids):
            if _parent_id(item) in anchors:
                adjacent.update(_parent_id(b) for b in route.block_ids[max(0, i - 1):i + 2])
    wait_related: dict[str, float] = {}
    if wait is not None:
        for op in schedule.operations:
            if op.robot_id == wait.robot_id or op.block_id not in blocks:
                continue
            overlap = max(0.0, min(op.end_time, wait.end_time) - max(op.start_time, wait.start_time))
            if overlap:
                distance = min(math.dist(wait.start, op.start), math.dist(wait.start, op.end))
                p = _parent_id(op.block_id)
                wait_related[p] = max(wait_related.get(p, 0.0), overlap / wait.duration / (0.5 + distance))
    related = []
    for p, bs in parent_blocks.items():
        if p in anchors:
            continue
        distance = min((math.dist(x, y) for b in bs for a in anchor_blocks
                        for x in (b.start, b.end) for y in (a.start, a.end)), default=20.0)
        receiver = any(owners[b.block_id] in receiver_robots for b in bs)
        weight = (0.1 + (1.0 + movable[p] + receiver) / (0.5 + distance)
                  + 2.0 * (p in adjacent) + 8.0 * wait_related.get(p, 0.0))
        related.append((p, weight))
    # Weighted sampling without replacement gives different anchors/context
    # across attempts, rather than repeatedly taking a deterministic top-q.
    ranked = sorted(related, key=lambda row: -math.log(max(rng.random(), 1e-15)) / row[1])
    removed = anchors[:q] + [p for p, _ in ranked[:max(0, q - len(anchors))]]
    return _remove_parent_set(current, removed)


def select_schedule_blocker_parents(current, schedule, evidence, *, q, seed, config):
    """Experimental direct-blocker core plus immediate route neighbours only.

    Returns the closed partial and selected evidence. q is an upper bound.
    Forbidden-interval union coverage ranks direct blocking parents; it is not
    a causal decomposition of waiting time. No distance/time proxy is added.
    """
    if not schedule.feasible or not 1<=q<=len(current.parents):
        raise ValueError('blocker selection requires a feasible schedule and valid q')
    blocks=block_map(current,config)
    events=[event for event in evidence if event.waiting.duration>config.numeric_epsilon
            and event.delayed_block_id in blocks
            and any(block_id in blocks for _,block_id,_ in event.blocker_intervals)]
    if not events:return None,None
    completion=schedule.robot_completion
    maximum=max(max(completion),config.numeric_epsilon)
    weights=[e.waiting.duration*(0.25+completion[e.waiting.robot_id]/maximum)**2 for e in events]
    event=random.Random(seed).choices(events,weights=weights,k=1)[0]
    delayed=blocks[event.delayed_block_id].parent_id
    intervals={}
    for _,block_id,spans in event.blocker_intervals:
        if block_id in blocks:
            intervals.setdefault(blocks[block_id].parent_id,[]).extend(spans)
    def contribution(parent):
        total=0.;end=-math.inf
        for lo,hi in sorted(intervals[parent]):
            total+=max(0.,hi-max(lo,end));end=max(end,hi)
        return total
    core=[delayed]+sorted((p for p in intervals if p!=delayed),key=lambda p:(-contribution(p),p))
    removed=core[:q]
    if len(removed)<q:
        neighbours=set()
        for route in current.routes:
            for i,block_id in enumerate(route.block_ids):
                if blocks[block_id].parent_id in core:
                    neighbours.update(blocks[b].parent_id for b in route.block_ids[max(0,i-1):i+2])
        removed+=sorted(neighbours-set(removed))[:q-len(removed)]
    return _remove_parent_set(current,removed),event


def _insert_one(
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    block: WeldingBlock,
    robot: int,
    position: int,
    hint: int,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...]]:
    new_routes = [list(route) for route in routes]
    new_hints = [list(vector) for vector in hints]
    new_routes[robot].insert(position, block.block_id)
    new_hints[robot].insert(position, hint)
    return (
        tuple(tuple(route) for route in new_routes),
        tuple(tuple(vector) for vector in new_hints),
    )


def _parent_alternatives(
    parent_id: str,
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    blocks: Mapping[str, WeldingBlock],
    pattern: SplitPattern,
    config: ScientificConfig,
    *,
    insertion_limit: int,
    remaining_budget: int,
    incremental_scoring: bool = False,
    direction_mode: str | None = None,
    original_directions: Mapping[str, int] | None = None,
    seed: int = 0,
    oriented_state: OrientedRouteState | None = None,
    coordination_proxy: bool = False,
    direction_versions: int = 1,
) -> tuple[tuple[_InsertionAlternative, ...], int]:
    if direction_mode is not None:
        return _oriented_parent_alternatives(
            parent_id, routes, hints, blocks, pattern, config,
            insertion_limit=insertion_limit, remaining_budget=remaining_budget,
            direction_mode=direction_mode, original_directions=original_directions or {},
            seed=seed, state=oriented_state, coordination_proxy=coordination_proxy,
            direction_versions=direction_versions,
        )
    parent_blocks = tuple(
        sorted(
            (block for block in blocks.values() if block.parent_id == parent_id),
            key=lambda block: block.block_id,
        )
    )
    base_proxy = _directed_proxy(routes, hints, blocks, config)
    if incremental_scoring:
        load_terms = tuple(
            tuple(config.process_time(blocks[item].length) for item in route)
            for route in routes
        )
        cached_loads = tuple(sum(terms) for terms in load_terms)
        arc_costs = tuple(
            _route_arc_costs(route, hints[robot], blocks, config)
            for robot, route in enumerate(routes)
        )
        # This assertion protects the tie-sensitive score ordering while the
        # optimized path is still an explicitly selected development option.
        assert _proxy_from_arc_costs(arc_costs) == base_proxy
    alternatives: list[_InsertionAlternative] = []
    evaluated = 0

    placements: list[tuple[int, int, int]] = []
    first = parent_blocks[0]
    first_robots = range(4)
    if pattern.kind is SplitKind.X_SPLIT:
        assert pattern.rail is not None
        first_robots = (0,) if pattern.rail.value == "UPPER" else (2,)
    for robot in first_robots:
        if not robot_is_eligible(first, robot, config):
            continue
        for position in bounded_insertion_positions(len(routes[robot]), insertion_limit):
            for hint in (0, 1):
                placements.append((robot, position, hint))

    for robot, position, hint in placements:
        if evaluated >= remaining_budget:
            break
        routes1, hints1 = _insert_one(routes, hints, first, robot, position, hint)
        if incremental_scoring:
            load_terms1 = list(load_terms)
            load_terms1[robot] = _insert_tuple(
                load_terms[robot], position, config.process_time(first.length)
            )
            loads1 = list(cached_loads)
            loads1[robot] = sum(load_terms1[robot])
            arc_costs1 = list(arc_costs)
            arc_costs1[robot] = _insert_route_arc_costs(
                routes[robot], hints[robot], arc_costs[robot], first,
                position, hint, blocks, config,
            )
        if len(parent_blocks) == 1:
            evaluated += 1
            if incremental_scoring:
                loads = tuple(loads1)
                delta = _proxy_from_arc_costs(arc_costs1) - base_proxy
            else:
                loads = _loads(routes1, blocks, config)
                delta = _directed_proxy(routes1, hints1, blocks, config) - base_proxy
            trace = ((first.block_id, robot, position, hint),)
            alternatives.append(
                _InsertionAlternative(
                    parent_id,
                    routes1,
                    hints1,
                    (max(loads), delta, robot, position, hint, trace),
                    trace,
                )
            )
            continue

        second = parent_blocks[1]
        second_robots = range(4)
        if pattern.kind is SplitKind.X_SPLIT:
            assert pattern.rail is not None
            second_robots = (1,) if pattern.rail.value == "UPPER" else (3,)
        for robot2 in second_robots:
            if evaluated >= remaining_budget:
                break
            if not robot_is_eligible(second, robot2, config):
                continue
            positions2 = bounded_insertion_positions(
                len(routes1[robot2]), insertion_limit
            )
            for position2 in positions2:
                if evaluated >= remaining_budget:
                    break
                for hint2 in (0, 1):
                    if evaluated >= remaining_budget:
                        break
                    # Insertion indices refer to different intermediate routes.
                    # Only final canonicalization may decide split collapse.
                    evaluated += 1
                    routes2, hints2 = _insert_one(
                        routes1, hints1, second, robot2, position2, hint2
                    )
                    if incremental_scoring:
                        load_terms2 = list(load_terms1)
                        load_terms2[robot2] = _insert_tuple(
                            load_terms1[robot2], position2,
                            config.process_time(second.length),
                        )
                        loads2 = list(loads1)
                        loads2[robot2] = sum(load_terms2[robot2])
                        arc_costs2 = list(arc_costs1)
                        arc_costs2[robot2] = _insert_route_arc_costs(
                            routes1[robot2], hints1[robot2], arc_costs1[robot2],
                            second, position2, hint2, blocks, config,
                        )
                        loads = tuple(loads2)
                        delta = _proxy_from_arc_costs(arc_costs2) - base_proxy
                    else:
                        loads = _loads(routes2, blocks, config)
                        delta = _directed_proxy(routes2, hints2, blocks, config) - base_proxy
                    trace = (
                        (first.block_id, robot, position, hint),
                        (second.block_id, robot2, position2, hint2),
                    )
                    alternatives.append(
                        _InsertionAlternative(
                            parent_id,
                            routes2,
                            hints2,
                            (
                                max(loads),
                                delta,
                                robot,
                                position,
                                hint,
                                robot2,
                                position2,
                                hint2,
                                trace,
                            ),
                            trace,
                        )
                    )
    return tuple(sorted(alternatives, key=lambda item: item.score)), evaluated


def oriented_route_state(routes, directions, blocks, config) -> OrientedRouteState:
    if len(routes) != 4 or len(directions) != 4:
        raise ValueError("four explicit routes and direction vectors are required")
    if any(len(route) != len(bits) or any(bit not in (0, 1) for bit in bits)
           for route, bits in zip(routes, directions)):
        raise ValueError("direction length/value mismatch")
    routes = tuple(tuple(route) for route in routes)
    directions = tuple(tuple(bits) for bits in directions)
    return OrientedRouteState(routes, directions, _loads(routes, blocks, config),
                              tuple(_route_arc_costs(route, directions[r], blocks, config)
                                    for r, route in enumerate(routes)))


def oriented_time_score(state: OrientedRouteState) -> tuple[float, float]:
    """Construction proxy only: maximum P+M, then total P+M.

    Rounding below a microsecond gives the secondary term a stable numerical
    tie break. Actual acceptance remains entirely based on B32 Cmax.
    """
    estimated = tuple(p + sum(arcs) for p, arcs in zip(state.loads, state.arc_costs))
    return round(max(estimated, default=0.0), 8), sum(estimated)


def oriented_route_lower_bound(state: OrientedRouteState) -> float:
    """Unrounded P+MOVE bound for these exact routes and directions.

    Routes start at their first weld and do not return home. Processing includes
    SETUP/WELD/POST through config.process_time; B32 can only add nonnegative
    coordination waits. The rounded construction score is deliberately unused.
    """
    return max((p + sum(arcs) for p, arcs in zip(state.loads, state.arc_costs)), default=0.0)


_NOMINAL_ROBOT_PAIRS = tuple(itertools.combinations(range(4), 2))
_Intervals = tuple[tuple[float, float], ...]


def _nominal_route_templates(state, robot, blocks, config):
    route = RobotRoute(robot, tuple(blocks[b] for b in state.routes[robot]), state.directions[robot])
    return build_operation_templates(route, config)


def _nominal_pair_intervals(first_ops, second_ops, config, *, same_rail):
    proximity, order = [], []
    eps = config.numeric_epsilon
    i = j = 0
    # Each route is contiguous and ordered: sweep overlapping phases.
    while i < len(first_ops) and j < len(second_ops):
        a, b = first_ops[i], second_ops[j]
        lo, hi = max(a.start_time, b.start_time), min(a.end_time, b.end_time)
        if hi > lo + eps:
            pa, pb = a.point_at(lo), b.point_at(lo)
            va = tuple((y - x) / a.duration for x, y in zip(a.start, a.end))
            vb = tuple((y - x) / b.duration for x, y in zip(b.start, b.end))
            near = (0.0, hi - lo)
            for axis, limit in ((0, config.interference_dx), (1, config.interference_dy)):
                near = _axis_band_interval(va[axis] - vb[axis], pa[axis] - pb[axis], limit + eps, *near)
                if near is None or near[1] <= near[0]:
                    near = None
                    break
            if near is not None:
                proximity.append((lo + near[0], lo + near[1]))
            if same_rail:
                # The first robot is the left robot (0 or 2).
                d0 = pa[0] + config.interference_dx - pb[0] - eps
                velocity = va[0] - vb[0]
                d1 = d0 + velocity * (hi - lo)
                if max(d0, d1) > 0.0:
                    start, end = lo, hi
                    if d0 <= 0.0:
                        start = lo - d0 / velocity
                    elif d1 <= 0.0:
                        end = lo - d0 / velocity
                    order.append((start, end))
        if a.end_time <= b.end_time:
            i += 1
        else:
            j += 1
    return tuple(proximity), tuple(order)


def _nominal_union_seconds(intervals):
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


@dataclass(frozen=True)
class NominalRouteCoordination:
    """Operation templates and pair exposures for a partial directed route set."""

    templates: tuple[tuple[Operation, ...], ...]
    pair_intervals: tuple[tuple[_Intervals, _Intervals], ...]

    @classmethod
    def from_state(cls, state: OrientedRouteState, blocks, config):
        templates = tuple(_nominal_route_templates(state, robot, blocks, config) for robot in range(4))
        intervals = tuple(_nominal_pair_intervals(templates[a], templates[b], config, same_rail=a // 2 == b // 2)
                          for a, b in _NOMINAL_ROBOT_PAIRS)
        return cls(templates, intervals)

    def with_changed_route(self, state: OrientedRouteState, robot: int, blocks, config):
        """Update the one route changed by an insertion, including neighbor flips."""
        route = _nominal_route_templates(state, robot, blocks, config)
        templates = self.templates[:robot] + (route,) + self.templates[robot + 1:]
        return self._with_templates(templates, (robot,), config)

    def with_changed_routes(self, state: OrientedRouteState, robots: tuple[int, ...], blocks, config):
        """For a split action, sweep each affected pair once on the final prefix."""
        if len(robots) == 1:
            return self.with_changed_route(state, robots[0], blocks, config)
        templates = tuple(_nominal_route_templates(state, robot, blocks, config) if robot in robots else previous
                          for robot, previous in enumerate(self.templates))
        return self._with_templates(templates, robots, config)

    def _with_templates(self, templates, robots, config):
        intervals = tuple(
            _nominal_pair_intervals(templates[a], templates[b], config, same_rail=a // 2 == b // 2)
            if a in robots or b in robots else previous
            for (a, b), previous in zip(_NOMINAL_ROBOT_PAIRS, self.pair_intervals)
        )
        return NominalRouteCoordination(templates, intervals)

    def features(self):
        """Union exposures per robot on simultaneous, no-wait timelines."""
        completion = tuple(ops[-1].end_time if ops else 0.0 for ops in self.templates)
        proximity = [[] for _ in range(4)]
        order = [[] for _ in range(4)]
        for (a, b), (near, crossing) in zip(_NOMINAL_ROBOT_PAIRS, self.pair_intervals):
            proximity[a].extend(near)
            proximity[b].extend(near)
            order[a].extend(crossing)
            order[b].extend(crossing)
        exposure = tuple(_nominal_union_seconds(near + crossing) for near, crossing in zip(proximity, order))
        return dict(nominal_completion=completion, nominal_cmax=max(completion),
                    total_nominal_time=sum(completion), exposure_seconds=exposure,
                    proximity_seconds=tuple(_nominal_union_seconds(v) for v in proximity),
                    same_rail_order_seconds=tuple(_nominal_union_seconds(v) for v in order),
                    exposure_score=max(t + e for t, e in zip(completion, exposure)))


def nominal_coordination_features(solution, directions, config):
    """Evaluate the nominal ranking features for a complete directed solution."""
    from mrta_reference.scope import FORMAL_SCOPE_V2

    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    state = oriented_route_state(tuple(r.block_ids for r in solution.routes), directions, blocks, config)
    return NominalRouteCoordination.from_state(state, blocks, config).features()


def oriented_insert_variants(
    state: OrientedRouteState, block: WeldingBlock, robot: int, position: int,
    blocks: Mapping[str, WeldingBlock], config: ScientificConfig, *,
    fixed_direction: int | None = None, locked_directions: Mapping[str, int] | None = None,
    max_evaluations: int = 8, keep: int = 2, seed: int = 0,
    check_initial_feasibility: bool = True,
) -> tuple[tuple[tuple[OrientedRouteState, OrientedInsertionAction], ...], int]:
    """One (block, robot, position, direction) decision with adjacent flips.

    Only arcs incident to the inserted block or its immediate neighbours are
    recomputed. Other routes and non-adjacent directions are retained verbatim.
    No route DP, scheduler or certifier is called here.
    Disabling the start check is only for a non-executed partial repair state;
    its caller must validate the completed start before returning a solution.
    """
    locked = locked_directions or {}
    route = state.routes[robot]
    if not 0 <= position <= len(route) or not robot_is_eligible(block, robot, config):
        raise ValueError("ineligible robot or invalid insertion position")
    new_route = route[:position] + (block.block_id,) + route[position:]
    original = dict(zip(route, state.directions[robot]))
    local = tuple(i for i in (position - 1, position, position + 1) if 0 <= i < len(new_route))
    choices = []
    for i in local:
        item = new_route[i]
        if item in locked:
            choices.append((locked[item],))
        elif fixed_direction is not None:
            choices.append((fixed_direction if i == position else original[item],))
        else:
            choices.append((0, 1))
    combinations = list(itertools.product(*choices))
    random.Random(seed).shuffle(combinations)
    # A local direction combination changes only this robot's vector. Keep
    # immutable route data once, and reuse identical geometry calculations.
    routes = state.routes[:robot] + (new_route,) + state.routes[robot + 1:]
    base_vector = list(state.directions[robot])
    base_vector.insert(position, 0)
    insertion_index = local.index(position)
    duration = config.process_time(block.length)
    loads = tuple(value + (duration if r == robot else 0.0)
                  for r, value in enumerate(state.loads))
    base_costs = _insert_route_arc_costs(route, state.directions[robot], state.arc_costs[robot],
                                         block, position, 0, blocks, config)
    refreshed_arcs = tuple(range(max(0, position - 2), min(len(new_route) - 1, position + 2)))
    arc_cache = {}
    other_points = ({r: oriented_endpoints(blocks[items[0]], state.directions[r][0])[0]
                     for r, items in enumerate(state.routes) if r != robot and items}
                    if check_initial_feasibility else {})
    first_block = blocks[new_route[0]]
    legal_first = {}
    route_totals = tuple(sum(arcs) for arcs in state.arc_costs)
    rows = []
    used = 0
    for bits in combinations[:max(0, max_evaluations)]:
        used += 1
        hint = bits[insertion_index]
        vector = base_vector.copy()
        for i, bit in zip(local, bits):
            vector[i] = bit
        if check_initial_feasibility:
            first = vector[0]
            if first not in legal_first:
                points = dict(other_points)
                points[robot] = oriented_endpoints(first_block, first)[0]
                legal_first[first] = _initial_error(points, config) is None
            if not legal_first[first]:
                continue
        directions = state.directions[:robot] + (tuple(vector),) + state.directions[robot + 1:]
        # Reuse old arcs, splice the insertion, then refresh only up to four
        # arcs touched by the predecessor/new block/successor direction tuple.
        costs = list(base_costs)
        for i in refreshed_arcs:
            key = (i, vector[i], vector[i + 1])
            if key not in arc_cache:
                _, end = oriented_endpoints(blocks[new_route[i]], vector[i])
                start, _ = oriented_endpoints(blocks[new_route[i + 1]], vector[i + 1])
                arc_cache[key] = math.dist(end, start) / config.empty_speed
            costs[i] = arc_cache[key]
        arcs = state.arc_costs[:robot] + (tuple(costs),) + state.arc_costs[robot + 1:]
        changes = tuple((new_route[i], original[new_route[i]], vector[i]) for i in local
                        if i != position and original[new_route[i]] != vector[i])
        action = OrientedInsertionAction(block.block_id, robot, position, hint, changes)
        candidate = OrientedRouteState(routes, directions, loads, arcs)
        rows.append((candidate, action))
    # Retain the original within-route and then robot-order summations exactly.
    rows.sort(key=lambda row: (sum(route_totals[:robot] + (sum(row[0].arc_costs[robot]),)
                                   + route_totals[robot + 1:]), row[0].directions))
    return tuple(rows[:keep]), used


def joint_action_key(action: OrientedInsertionAction) -> tuple:
    return (action.block_id, action.robot, action.position, action.direction,
            tuple(tuple(change) for change in action.neighbour_changes))


def joint_block_robots(block, pattern, config):
    """Eligible owners for a block, with spatial ownership for formal X splits."""
    if pattern.kind is SplitKind.X_SPLIT:
        return ((0 if pattern.rail.value == "UPPER" else 2) + int(block.block_id.endswith("::1")),)
    return tuple(r for r in range(4) if robot_is_eligible(block, r, config))


def _joint_insertion_locations(state, block, pattern, blocks, config, insertion_limit, robot_id=None):
    """Common robot/position pool, including direction-independent sibling pruning."""
    for robot in joint_block_robots(block, pattern, config):
        if robot_id is not None and robot != robot_id:
            continue
        route = state.routes[robot]
        for position in bounded_insertion_positions(len(route), insertion_limit):
            if any(blocks[route[i]].parent_id == block.parent_id and route[i] != block.block_id
                   for i in (position - 1, position) if 0 <= i < len(route)):
                continue
            yield robot, position


def has_joint_block_candidate(state, block, pattern, blocks, config, *,
                              insertion_limit=8, check_initial_feasibility=True, robot_id=None):
    """Exact membership of the joint action pool, without building route costs.

    The public generator evaluates every local direction tuple (at most eight)
    and retains all eight. At position 0 the new head has both directions; at
    position 1 the previous head may flip; later positions leave it unchanged.
    Other robots' heads stay fixed. Arc costs and sorting cannot remove the
    last legal tuple, so the same initial-head check decides existence.
    """
    points = None
    unchanged_legal = None
    legal_heads = {}
    for robot, position in _joint_insertion_locations(state, block, pattern, blocks, config, insertion_limit, robot_id):
        # Match oriented_insert_variants' validation for malformed forced X
        # assignments, after the same position/sibling filtering.
        if not robot_is_eligible(block, robot, config):
            raise ValueError("ineligible robot or invalid insertion position")
        if not check_initial_feasibility:
            return True
        if points is None:
            points = {r: oriented_endpoints(blocks[route[0]], state.directions[r][0])[0]
                      for r, route in enumerate(state.routes) if route}
        if position >= 2:
            if unchanged_legal is None:
                unchanged_legal = _initial_error(points, config) is None
            if unchanged_legal:
                return True
            continue
        head = block if position == 0 else blocks[state.routes[robot][0]]
        key = robot, head.block_id
        if key not in legal_heads:
            legal_heads[key] = False
            for direction in (0, 1):
                candidate_points = dict(points)
                candidate_points[robot] = oriented_endpoints(head, direction)[0]
                if _initial_error(candidate_points, config) is None:
                    legal_heads[key] = True
                    break
        if legal_heads[key]:
            return True
    return False


def joint_block_candidates(
    state: OrientedRouteState, block: WeldingBlock, pattern: SplitPattern,
    blocks: Mapping[str, WeldingBlock], config: ScientificConfig, *,
    insertion_limit: int = 8, first_only: bool = False, check_initial_feasibility: bool = True,
    prefer_legal_start_recovery: bool = True, robot_id: int | None = None,
):
    """Shared student/teacher-replay action set, including neighbour flips.

    All at-most-eight local direction configurations are exposed. The existing
    bounded position rule is identical at training and inference. No teacher
    label or future schedule is accepted by this function.
    """
    rows = []
    for robot, position in _joint_insertion_locations(state, block, pattern, blocks, config, insertion_limit, robot_id):
        variants, _ = oriented_insert_variants(state, block, robot, position, blocks, config,
                                               max_evaluations=8, keep=8,
                                               check_initial_feasibility=check_initial_feasibility)
        for inserted, action in variants:
            rows.append((inserted, action))
            if first_only:
                return tuple(rows)
    if not check_initial_feasibility and prefer_legal_start_recovery:
        rows=prefer_joint_legal_starts(rows,blocks,config)
    return tuple(rows)


def prefer_joint_legal_starts(variants, blocks, config):
    """Existing initial-head preference, also applied after fixed-bit filtering."""
    restored=[]
    for inserted,action in variants:
        points={r:oriented_endpoints(blocks[route[0]],inserted.directions[r][0])[0]
                for r,route in enumerate(inserted.routes) if route}
        if _initial_error(points,config) is None:
            restored.append((inserted,action))
    return tuple(restored or variants)


def select_feedback_child(current, feedback, config, *, seed=None):
    """A movable block on the actual B32 bottleneck, without sibling removal."""
    blocks = block_map(current, config)
    patterns = {p.parent_id: p for p in current.patterns}
    critical = max(range(4), key=lambda r: (feedback['completion'][r], -r))
    candidates = [b for b in current.routes[critical].block_ids if any(
        r != critical for r in joint_block_robots(blocks[b], patterns[blocks[b].parent_id], config))]
    contribution = lambda b: feedback['block_processing'][b] + feedback['block_move'][b]
    ordered = sorted(candidates, key=lambda b: (-contribution(b), b))
    if not ordered:
        return None
    if seed is None:
        return ordered[0]
    # Vary among the four largest observed contributors; no geometric WAIT.
    pool = ordered[:4]
    return random.Random(seed).choices(pool, weights=[max(contribution(b), 1e-9) for b in pool])[0]


def child_transfer_candidates(current, directions, block_id, config, *, destination_robot=None,
                              insertion_limit=8):
    """INTER_RELOCATE one block with joint position/direction/neighbour choices.

    A sibling keeps its robot, route index and direction. Canonicalization may
    merge consecutive same-robot children; such a pattern change is rejected,
    including a merge of an unrelated parent exposed by removing this block.
    """
    from mrta_reference.candidate import apply_candidate
    from mrta_reference.scope import FORMAL_SCOPE_V2

    blocks = block_map(current, config, scope=FORMAL_SCOPE_V2)
    if block_id not in blocks:
        raise ValueError('unknown child block')
    block = blocks[block_id]
    pattern = next(p for p in current.patterns if p.parent_id == block.parent_id)
    owners = {b: (r, i, directions[r][i]) for r, route in enumerate(current.routes)
              for i, b in enumerate(route.block_ids)}
    source, position, _ = owners[block_id]
    destinations = tuple(r for r in joint_block_robots(block, pattern, config)
                         if r != source and (destination_robot is None or r == destination_robot))
    if not destinations:
        return ()
    routes = tuple(tuple(b for b in route.block_ids if b != block_id) for route in current.routes)
    bits = tuple(tuple(d for b, d in zip(route.block_ids, vector) if b != block_id)
                 for route, vector in zip(current.routes, directions))
    partial = oriented_route_state(routes, bits, blocks, config)
    siblings = {b for b, value in blocks.items() if value.parent_id == block.parent_id and b != block_id}
    answers = []
    for destination in destinations:
        variants = joint_block_candidates(partial, block, pattern, blocks, config,
            insertion_limit=insertion_limit, robot_id=destination, check_initial_feasibility=True)
        by_position = {}
        for after, action in variants:
            if action.position not in by_position:
                move = CandidateMove(CandidateKey(current.revision, MoveType.INTER_RELOCATE,
                    (block.parent_id,), source, destination, (position,), (action.position,)), (block_id,))
                try:
                    solution = apply_candidate(current, move, config, scope=FORMAL_SCOPE_V2)
                except ValueError:
                    solution = None
                if solution is not None and (solution.patterns != current.patterns or
                        tuple(r.block_ids for r in solution.routes) != after.routes):
                    solution = None
                by_position[action.position] = (move, solution)
            move, solution = by_position[action.position]
            if solution is None:
                continue
            after_owners = {b: (r, i, after.directions[r][i]) for r, route in enumerate(after.routes)
                            for i, b in enumerate(route)}
            if any(after_owners[b] != owners[b] for b in siblings):
                continue
            # The primitive can only flip immediate destination neighbours.
            if any(after.routes[r] != routes[r] or after.directions[r] != bits[r]
                   for r in range(4) if r != destination):
                raise ValueError('child insertion changed an unrelated route')
            answers.append(ChildTransfer(move, action, partial, after, solution))
    return tuple(answers)


def _oriented_parent_alternatives(
    parent_id, routes, hints, blocks, pattern, config, *, insertion_limit,
    remaining_budget, direction_mode, original_directions, seed, state=None,
    coordination_proxy=False, direction_versions=1,
):
    state = state or oriented_route_state(routes, hints, blocks, config)
    children = sorted((b for b in blocks.values() if b.parent_id == parent_id), key=lambda b: b.block_id)
    rng = random.Random(seed)
    legal = []
    for index, child in enumerate(children):
        robots = [r for r in range(4) if robot_is_eligible(child, r, config)]
        if pattern.kind is SplitKind.X_SPLIT:
            robots = [(0 if pattern.rail.value == "UPPER" else 2) + index]
        legal.append(robots)
    groups = list(itertools.product(*legal))
    rng.shuffle(groups)
    alternatives = []
    evaluated = 0
    for group_index, robots in enumerate(groups):
        # Every legal robot (or child robot pair) receives its own quota. With
        # very small budgets the seeded group rotation avoids persistent R0 bias.
        quota = remaining_budget // len(groups) + (group_index < remaining_budget % len(groups))
        used_group = 0
        positions = [bounded_insertion_positions(len(routes[robots[0]]), insertion_limit)]
        if len(children) == 2:
            positions.append(bounded_insertion_positions(len(routes[robots[1]]) + int(robots[0] == robots[1]), insertion_limit))
        plans = list(itertools.product(*positions))
        rng.shuffle(plans)
        for plan in plans:
            left = quota - used_group
            if left < len(children):
                break
            allowance = min(left, 8 if len(children) == 1 else 24)
            first_budget = allowance if len(children) == 1 else max(1, allowance // 3)
            first_rows, used = oriented_insert_variants(
                state, children[0], robots[0], plan[0], blocks, config,
                fixed_direction=(original_directions[children[0].block_id] if direction_mode == "FIXED" else None),
                max_evaluations=first_budget, keep=2, seed=rng.randrange(2**32),
            )
            used_group += used
            complete = [(s, (a,)) for s, a in first_rows]
            if len(children) == 2:
                complete = []
                for j, (first_state, first_action) in enumerate(first_rows):
                    second_budget = (allowance - (used_group - (quota - left))) // (len(first_rows) - j)
                    second_rows, used = oriented_insert_variants(
                        first_state, children[1], robots[1], plan[1], blocks, config,
                        fixed_direction=(original_directions[children[1].block_id] if direction_mode == "FIXED" else None),
                        max_evaluations=max(0, second_budget), keep=2, seed=rng.randrange(2**32),
                    )
                    used_group += used
                    complete.extend((s, (first_action, a)) for s, a in second_rows)
            for final, actions in complete:
                # Adjacent children would be merged by canonicalize. Exclude
                # this structural action before it can crowd out fixed-mode alternatives.
                if len(children) == 2 and any(
                    children[0].block_id in route and children[1].block_id in route
                    and abs(route.index(children[0].block_id) - route.index(children[1].block_id)) == 1
                    for route in final.routes
                ):
                    continue
                trace = tuple((a.block_id, a.robot, a.position, a.direction) for a in actions)
                proxy = (oriented_time_score(final) if coordination_proxy
                         else (max(final.loads), final.travel - state.travel))
                score = (*proxy, final.routes, final.directions)
                alternatives.append(_InsertionAlternative(parent_id, final.routes, final.directions,
                                                           score, trace, actions, final))
        evaluated += used_group
    # Regret compares distinct route structures, never two direction variants
    # of the same assignment and insertion location.
    by_structure = {}
    for alternative in sorted(alternatives, key=lambda a: a.score):
        versions = by_structure.setdefault(alternative.routes, [])
        if len(versions) < direction_versions and all(a.hints != alternative.hints for a in versions):
            versions.append(alternative)
    return tuple(sorted((a for versions in by_structure.values() for a in versions),
                        key=lambda a: a.score)), evaluated


def repair_partial_state(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    partial: PartialSearchState,
    destroy_operator: DestroyOperator,
    repair_operator: RepairOperator,
    config: ScientificConfig,
    *,
    insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096,
    incremental_scoring: bool = False,
    preserve_patterns: bool = False,
    direction_mode: str | None = None,
    seed: int = 0,
    coordination_proxy: bool = False,
    direction_versions: int = 1,
    allow_pattern_changes: bool = False,
) -> RepairResult:
    if partial.source_revision != current.revision:
        raise ValueError("stale partial state revision")
    if max_insertion_evaluations < 1:
        raise ValueError("repair insertion budget must be positive")
    if direction_versions not in (1, 2):
        raise ValueError("at most two direction versions are supported")
    if direction_mode is None and (coordination_proxy or direction_versions != 1):
        raise ValueError("oriented repair options require explicit direction state")
    patterns = {pattern.parent_id: pattern for pattern in partial.patterns}
    current_blocks = block_map(current, config)
    if allow_pattern_changes:
        from mrta_reference.scope import FORMAL_SCOPE_V2
        if partial.parents != current.parents or direction_mode != "JOINT":
            raise ValueError("mode repair requires the same parents and joint directions")
        old_patterns = {p.parent_id: p for p in current.patterns}
        pending_ids = set(partial.removed_parent_ids)
        if any(p != old_patterns[p.parent_id] and p.parent_id not in pending_ids
               for p in partial.patterns):
            raise ValueError("mode changes require complete removal of the parent")
        retained = [b for route in partial.routes for b in route]
        expected = {b for b, block in current_blocks.items() if block.parent_id not in pending_ids}
        if len(retained) != len(set(retained)) or set(retained) != expected:
            raise ValueError("mode repair requires parent-closed partial routes")
        blocks = block_map(partial, config, scope=FORMAL_SCOPE_V2)
    else:
        blocks = current_blocks
    current_hints_by_block = _hints_from_current(current, current_directions)
    if direction_mode is not None:
        if direction_mode not in ("FIXED", "JOINT"):
            raise ValueError("unknown oriented repair mode")
        oriented_route_state(tuple(r.block_ids for r in current.routes), current_directions, current_blocks, config)
    routes = partial.routes
    hints = tuple(
        tuple(current_hints_by_block.get(item, 0) for item in route)
        for route in routes
    )
    oriented_state = (oriented_route_state(routes, hints, blocks, config)
                      if direction_mode is not None else None)
    pending = list(partial.removed_parent_ids)
    parents = {parent.parent_id: parent for parent in current.parents}
    trace: list[tuple[str, tuple[tuple[str, int, int, int], ...]]] = []
    evaluations = 0
    actions: list[OrientedInsertionAction] = []
    rng = random.Random(seed)
    direction_paths = ((oriented_state, ()),)

    while pending:
        if repair_operator is RepairOperator.GREEDY_REPAIR:
            selected_parent = min(
                pending,
                key=lambda parent_id: (
                    -config.process_time(parents[parent_id].length),
                    parent_id,
                ),
            )
            parents_to_evaluate = (selected_parent,)
        else:
            parents_to_evaluate = tuple(sorted(pending))
        alternatives_by_parent: dict[str, tuple[_InsertionAlternative, ...]] = {}
        future_calls = max(
            1,
            len(pending)
            if repair_operator is RepairOperator.GREEDY_REPAIR
            else len(pending) * (len(pending) + 1) // 2,
        )
        per_call_budget = max(
            1, (max_insertion_evaluations - evaluations) // future_calls
        )
        for parent_id in parents_to_evaluate:
            allowance = min(per_call_budget, max_insertion_evaluations - evaluations)
            sources = direction_paths if direction_versions == 2 else ((oriented_state, ()),)
            combined = []
            for source_index, (source, prefix_actions) in enumerate(sources):
                source_budget = allowance // len(sources) + (source_index < allowance % len(sources))
                if source_budget < 1:
                    continue
                oriented_options = ({} if direction_mode is None else dict(
                    direction_mode=direction_mode, original_directions=current_hints_by_block,
                    seed=rng.randrange(2**32), oriented_state=source,
                    coordination_proxy=coordination_proxy, direction_versions=direction_versions,
                ))
                alternatives, used = _parent_alternatives(
                    parent_id, routes, (source.directions if source is not None else hints),
                    blocks, patterns[parent_id], config,
                    insertion_limit=insertion_limit, remaining_budget=source_budget,
                    incremental_scoring=incremental_scoring, **oriented_options,
                )
                evaluations += used
                if direction_versions == 2:
                    alternatives = tuple(replace(a, actions=prefix_actions + a.actions) for a in alternatives)
                combined.extend(alternatives)
            alternatives = tuple(sorted(combined, key=lambda a: a.score))
            alternatives_by_parent[parent_id] = alternatives
            if evaluations >= max_insertion_evaluations:
                break
        viable = {
            parent_id: alternatives
            for parent_id, alternatives in alternatives_by_parent.items()
            if alternatives
        }
        if not viable:
            return RepairResult(None, evaluations, "NO_LEGAL_REPAIR_INSERTION")
        if repair_operator is RepairOperator.GREEDY_REPAIR:
            selected_parent = next(iter(viable))
        else:
            regret_rows = []
            for parent_id, alternatives in viable.items():
                best = alternatives[0]
                second = (next((a for a in alternatives[1:] if a.routes != best.routes), None)
                          if direction_versions == 2 else (alternatives[1] if len(alternatives) > 1 else None))
                primary = (
                    math.inf
                    if second is None
                    else float(second.score[0]) - float(best.score[0])
                )
                secondary = (
                    math.inf
                    if second is None
                    else float(second.score[1]) - float(best.score[1])
                )
                regret_rows.append(
                    (
                        -primary,
                        -secondary,
                        -config.process_time(parents[parent_id].length),
                        parent_id,
                    )
                )
            selected_parent = min(regret_rows)[3]
        selected = viable[selected_parent][0]
        routes, hints = selected.routes, selected.hints
        oriented_state = selected.oriented_state
        trace.append((selected_parent, selected.trace))
        if direction_versions == 2:
            actions = list(selected.actions)
            retained = []
            for alternative in viable[selected_parent]:
                if alternative.routes == selected.routes and all(alternative.hints != p[0].directions for p in retained):
                    retained.append((alternative.oriented_state, alternative.actions))
                    if len(retained) == direction_versions:
                        break
            direction_paths = tuple(retained)
        else:
            actions.extend(selected.actions)
        pending.remove(selected_parent)

    try:
        solution = canonicalize(
            partial.parents,
            partial.patterns,
            tuple(Route(robot, routes[robot]) for robot in range(4)),
            config,
            revision=current.revision + 1,
            x_split_validator=(
                finite_x_split_validator(current.parents, config)
                if any(pattern.kind is SplitKind.X_SPLIT for pattern in partial.patterns)
                else None
            ),
        )
        solution_blocks = block_map(solution, config)
        target_patterns = partial.patterns if allow_pattern_changes else current.patterns
        if preserve_patterns and solution.patterns != target_patterns:
            return RepairResult(None, evaluations, "FIXED_PATTERN_CHANGED_BY_CANONICALIZATION")
        if any(
            not robot_is_eligible(solution_blocks[item], route.robot_id, config)
            for route in solution.routes
            for item in route.block_ids
        ):
            raise ValueError("robot eligibility failure")
    except (ValueError, KeyError, IndexError) as error:
        return RepairResult(None, evaluations, f"CANONICAL_REPAIR_FAILURE:{error}")
    current_blocks = block_map(current, config)
    solution_hints_by_block = {
        item: hints[robot][index]
        for robot, route in enumerate(routes)
        for index, item in enumerate(route)
    }
    final_routes = tuple(route.block_ids for route in solution.routes)
    final_hints = tuple(
        tuple(solution_hints_by_block.get(item, 0) for item in route)
        for route in final_routes
    )
    current_routes = tuple(route.block_ids for route in current.routes)
    current_hints = tuple(
        tuple(current_hints_by_block.get(item, 0) for item in route)
        for route in current_routes
    )
    current_loads = _loads(current_routes, current_blocks, config)
    final_loads = _loads(final_routes, solution_blocks, config)
    if direction_versions == 2:
        parent_order = tuple(dict.fromkeys(_parent_id(a.block_id) for a in actions))
        trace = [(p, tuple((a.block_id, a.robot, a.position, a.direction) for a in actions
                           if _parent_id(a.block_id) == p)) for p in parent_order]
    repair_identity = tuple(trace)
    if allow_pattern_changes:
        repair_identity = (tuple((p.pattern_id, p.t) for p in partial.patterns), repair_identity)
    identity = CompleteCandidateIdentity(
        current.revision,
        CandidateSourceKind.LNS_REPAIRED.value,
        (
            destroy_operator.value,
            repair_operator.value,
            partial.removed_parent_ids,
            repair_identity,
        ),
    )
    provenance = (
        ("destroy_operator", destroy_operator.value),
        ("repair_operator", repair_operator.value),
        ("removed_parent_ids", partial.removed_parent_ids),
        ("q", len(partial.removed_parent_ids)),
        ("repair_trace", repair_identity),
    )
    candidate = CompleteSearchCandidate(
        CandidateSourceKind.LNS_REPAIRED,
        solution,
        identity,
        max(final_loads),
        _directed_proxy(final_routes, final_hints, solution_blocks, config)
        - _directed_proxy(current_routes, current_hints, current_blocks, config),
        _optional_split_count(solution, config) - _optional_split_count(current, config),
        (max(final_loads) - min(final_loads))
        - (max(current_loads) - min(current_loads)),
        provenance,
    )
    return RepairResult(candidate, evaluations, directions=(final_hints if direction_mode is not None else None),
                        actions=tuple(actions), direction_candidates=(
                            tuple(p[0].directions for p in direction_paths) if direction_versions == 2
                            else ((final_hints,) if direction_mode is not None else ())),
                        oriented_state=(oriented_state if oriented_state is not None
                            and oriented_state.routes == final_routes and oriented_state.directions == final_hints else None))


def repair_partial_state_oriented(
    current: CanonicalSolution, current_directions: DirectionVectors,
    partial: PartialSearchState, config: ScientificConfig, *,
    fixed_directions: bool = False, insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096, seed: int = 0,
    coordination_proxy: bool = False, direction_versions: int = 1,
    allow_pattern_changes: bool = False,
) -> RepairResult:
    """The existing Regret-2 loop with explicit, persistent action directions."""
    return repair_partial_state(
        current, current_directions, partial, DestroyOperator.RANDOM_REMOVAL,
        RepairOperator.REGRET_2_REPAIR, config, insertion_limit=insertion_limit,
        max_insertion_evaluations=max_insertion_evaluations, preserve_patterns=True,
        direction_mode="FIXED" if fixed_directions else "JOINT", seed=seed,
        coordination_proxy=coordination_proxy, direction_versions=direction_versions,
        allow_pattern_changes=allow_pattern_changes,
    )


def rank_parent_mode_changes(current, partial, config):
    """Rank one-parent legal mode changes by an assignment-only load estimate.

    Each alternative includes its actual per-block setup/post cost. Other
    parents keep their incumbent assignment only for this inexpensive ranking;
    the subsequent repair can move the entire removed set. This is neither a
    schedulability test nor a lower bound on that subsequent repair.
    """
    from mrta_reference.geometry import build_legal_pattern_catalog, blocks_for_pattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    catalog = build_legal_pattern_catalog(current.parents, config, FORMAL_SCOPE_V2)
    old = {p.parent_id: p for p in current.patterns}
    parents = {p.parent_id: p for p in current.parents}
    blocks = block_map(current, config, scope=FORMAL_SCOPE_V2)
    owners = {b: r for r, route in enumerate(current.routes) for b in route.block_ids}
    loads = _loads(tuple(r.block_ids for r in current.routes), blocks, config)
    rows = []
    for pid in partial.removed_parent_ids:
        residual = list(loads)
        for b, block in blocks.items():
            if block.parent_id == pid:
                residual[owners[b]] -= config.process_time(block.length)
        for pattern in catalog[pid]:
            if pattern == old[pid]:
                continue
            children = blocks_for_pattern(parents[pid], pattern, config)
            legal = []
            for child in children:
                if pattern.kind is SplitKind.X_SPLIT:
                    legal.append(((0 if pattern.rail.value == 'UPPER' else 2)
                                  + int(child.block_id.endswith('::1')),))
                else:
                    legal.append(tuple(r for r in range(4) if robot_is_eligible(child, r, config)))
            estimates = []
            for assignment in itertools.product(*legal):
                # Adjacent same-robot siblings normalize to WHOLE in the real
                # model. Same-robot split assignments remain a permissive proxy:
                # repair must insert them non-adjacently or choose WHOLE itself.
                projected = list(residual)
                for child, robot in zip(children, assignment):
                    projected[robot] += config.process_time(child.length)
                estimates.append((max(projected), max(projected)-min(projected)))
            if estimates:
                score = (*min(estimates), sum(config.process_time(c.length) for c in children),
                         pid, pattern.pattern_id)
                rows.append((score, pattern))
    return tuple(pattern for _, pattern in sorted(rows, key=lambda row: row[0]))


def rank_parent_mode_pairs(current, partial, config, *, limit=3, diverse_pairs=False):
    """Rank simultaneous changes of two distinct removed parents.

    Enumerate their full public legal mode catalogs, not just individually
    promising modes. Minimize max processing load, then spread and total new
    processing. All other parents keep incumbent owners for this proxy only;
    actual repair may move the entire removed set. No scheduler is called.
    With diverse_pairs, take the best mode of each distinct unordered parent
    pair first, then fill unused slots by the original ranking. The top proposal
    is unchanged; diversity is a candidate coverage heuristic, not a new score.
    """
    import numpy as np
    from mrta_reference.geometry import build_legal_pattern_catalog, blocks_for_pattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    if limit == 0 or len(partial.removed_parent_ids) < 2:
        return ()
    catalog = build_legal_pattern_catalog(current.parents, config, FORMAL_SCOPE_V2)
    old = {p.parent_id:p for p in current.patterns}
    parents = {p.parent_id:p for p in current.parents}
    blocks = block_map(current, config, scope=FORMAL_SCOPE_V2)
    routes = tuple(r.block_ids for r in current.routes)
    loads = np.asarray(_loads(routes, blocks, config))
    owners = {b:r for r,route in enumerate(routes) for b in route}
    options = {}; contributions = {}
    for pid in partial.removed_parent_ids:
        contributions[pid] = np.zeros(4)
        for b,block in blocks.items():
            if block.parent_id == pid:
                contributions[pid][owners[b]] += config.process_time(block.length)
        options[pid] = []
        for pattern in catalog[pid]:
            if pattern == old[pid]:
                continue
            children = blocks_for_pattern(parents[pid], pattern, config)
            legal = [((0 if pattern.rail.value == 'UPPER' else 2)+index,)
                     if pattern.kind is SplitKind.X_SPLIT else
                     tuple(r for r in range(4) if robot_is_eligible(child,r,config))
                     for index,child in enumerate(children)]
            vectors = []
            for assignment in itertools.product(*legal):
                vector = np.zeros(4)
                for child,robot in zip(children,assignment):
                    vector[robot] += config.process_time(child.length)
                vectors.append(vector)
            if vectors:
                options[pid].append((pattern,np.asarray(vectors),sum(config.process_time(c.length) for c in children)))
    rows = []
    for a,b in itertools.combinations(sorted(options),2):
        residual = loads-contributions[a]-contributions[b]
        for pa,va,work_a in options[a]:
            for pb,vb,work_b in options[b]:
                projected = (residual+va[:,None,:]+vb[None,:,:]).reshape(-1,4)
                peaks = projected.max(axis=1)
                spread = peaks-projected.min(axis=1)
                best = np.lexsort((spread,peaks))[0]
                score = (float(peaks[best]),float(spread[best]),work_a+work_b,
                         a,pa.pattern_id,pa.t if pa.t is not None else -1.,
                         b,pb.pattern_id,pb.t if pb.t is not None else -1.)
                rows.append((score,(pa,pb)))
    ordered=[patterns for _,patterns in sorted(rows,key=lambda row:row[0])]
    if not diverse_pairs:
        return tuple(ordered[:limit])
    chosen=[];seen_pairs=set()
    for patterns in ordered:
        key=tuple(p.parent_id for p in patterns)
        if key not in seen_pairs:
            chosen.append(patterns);seen_pairs.add(key)
            if len(chosen)==limit:return tuple(chosen)
    chosen.extend(p for p in ordered if p not in chosen)
    return tuple(chosen[:limit])


def _block_mid_x(block: WeldingBlock) -> float:
    return 0.5 * (block.start[0] + block.end[0])


def _parent_mid_x(parent_id: str, blocks: Mapping[str, WeldingBlock]) -> float:
    rows = [block for block in blocks.values() if block.parent_id == parent_id]
    return sum(_block_mid_x(block) for block in rows) / len(rows)


def _robot_by_block(solution: CanonicalSolution) -> dict[str, int]:
    return {
        block_id: robot
        for robot, route in enumerate(solution.routes)
        for block_id in route.block_ids
    }


def _weld_parent_neighbours(
    schedule: ScheduleResult, robot: int, wait_operation
) -> tuple[str, ...]:
    """Associate WAIT with adjacent real WELDs; this is not a causal claim."""
    welds = sorted(
        (
            operation
            for operation in schedule.operations
            if operation.robot_id == robot
            and operation.kind is OperationKind.WELD
            and operation.block_id is not None
        ),
        key=lambda operation: (
            operation.start_time,
            operation.end_time,
            operation.sequence_index,
            operation.operation_id,
        ),
    )
    before = [op for op in welds if op.end_time <= wait_operation.start_time + 1.0e-9]
    after = [op for op in welds if op.start_time >= wait_operation.end_time - 1.0e-9]
    adjacent = []
    if before:
        adjacent.append(before[-1])
    if after:
        adjacent.append(after[0])
    if not adjacent and welds:
        adjacent.append(min(welds, key=lambda op: (
            abs(op.sequence_index - wait_operation.sequence_index),
            abs(op.start_time - wait_operation.start_time), op.operation_id,
        )))
    return tuple(dict.fromkeys(_parent_id(op.block_id) for op in adjacent))


def identify_crrr_region(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *,
    attempt_index: int,
    min_parents: int = 2,
    max_parents: int = 6,
) -> CRRRRegion | None:
    """Locate one bounded schedule- and geometry-anchored parent region."""
    if attempt_index < 0 or attempt_index >= 8:
        raise ValueError("CRRR attempt_index must be in 0..7")
    if not (2 <= min_parents <= max_parents <= 6):
        raise ValueError("CRRR parent bounds must satisfy 2 <= min <= max <= 6")
    blocks = block_map(current, config)
    robot_by_block = _robot_by_block(current)
    critical_robot = max(
        range(4), key=lambda robot: (schedule.robot_completion[robot], -robot)
    )
    partner_robot = critical_robot ^ 1
    waits = sorted(
        (
            operation
            for operation in schedule.operations
            if operation.kind is OperationKind.WAIT
            and operation.duration > config.numeric_epsilon
        ),
        key=lambda operation: (
            -int(operation.robot_id == critical_robot), -operation.duration,
            operation.robot_id, operation.start_time, operation.operation_id,
        ),
    )
    use_wait = bool(waits) and attempt_index % 2 == 0
    anchors: list[str] = []
    wait_duration = 0.0
    if use_wait:
        wait = waits[(attempt_index // 2) % len(waits)]
        critical_robot = wait.robot_id
        partner_robot = critical_robot ^ 1
        wait_duration = wait.duration
        anchors.extend(_weld_parent_neighbours(schedule, critical_robot, wait))
        anchor_points = [
            _parent_mid_x(parent_id, blocks)
            for parent_id in anchors
            if any(block.parent_id == parent_id for block in blocks.values())
        ]
        anchor_x = (
            sum(anchor_points) / len(anchor_points)
            if anchor_points else 0.5 * (wait.start[0] + wait.end[0])
        )
        partner_welds = sorted(
            (
                operation
                for operation in schedule.operations
                if operation.robot_id == partner_robot
                and operation.kind is OperationKind.WELD
                and operation.block_id is not None
                and operation.start_time <= wait.end_time + 1.0e-9
                and operation.end_time >= wait.start_time - 1.0e-9
            ),
            key=lambda operation: (
                abs(0.5 * (operation.start[0] + operation.end[0]) - anchor_x),
                operation.sequence_index,
            ),
        )
        anchors.extend(_parent_id(operation.block_id) for operation in partner_welds[:2])
        bottleneck = CRRRBottleneck.WAIT_BOUNDARY
    else:
        left_robot = 0 if critical_robot < 2 else 2
        right_robot = left_robot + 1
        left_x = [_block_mid_x(blocks[item]) for item in current.routes[left_robot].block_ids]
        right_x = [_block_mid_x(blocks[item]) for item in current.routes[right_robot].block_ids]
        if left_x and right_x:
            anchor_x = 0.5 * (max(left_x) + min(right_x))
        else:
            rail_x = left_x + right_x
            anchor_x = sorted(rail_x)[len(rail_x) // 2] if rail_x else 0.0
        bottleneck = CRRRBottleneck.LOAD_BOUNDARY
    rail_parents = {
        blocks[item].parent_id
        for robot in (critical_robot, partner_robot)
        for item in current.routes[robot].block_ids
    }
    ordered = sorted(
        (parent.parent_id for parent in current.parents),
        key=lambda parent_id: (
            -int(parent_id in anchors), -int(parent_id in rail_parents),
            abs(_parent_mid_x(parent_id, blocks) - anchor_x),
            -int(any(
                robot_by_block[block.block_id] == critical_robot
                for block in blocks.values() if block.parent_id == parent_id
            )),
            parent_id,
        ),
    )
    target_count = min(max_parents, max(min_parents, 2 + attempt_index % 5))
    selected = tuple(sorted(ordered[:target_count]))
    if len(selected) < min_parents:
        return None
    return CRRRRegion(
        bottleneck, selected, critical_robot, partner_robot, anchor_x, wait_duration
    )


def _adjacency_by_parent(routes: Sequence[Sequence[str]]) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    for route in routes:
        parents = tuple(_parent_id(item) for item in route)
        for index, parent_id in enumerate(parents):
            neighbours = result.setdefault(parent_id, set())
            if index:
                neighbours.add((parents[index - 1], parent_id))
            if index + 1 < len(parents):
                neighbours.add((parent_id, parents[index + 1]))
    return result


def _route_overlap_proxy(
    routes: Sequence[Sequence[str]], blocks: Mapping[str, WeldingBlock]
) -> float:
    envelopes = []
    for route in routes:
        xs = [x for item in route for x in (blocks[item].start[0], blocks[item].end[0])]
        envelopes.append(None if not xs else (min(xs), max(xs)))
    total = 0.0
    for left, right in ((0, 1), (2, 3)):
        if envelopes[left] is not None and envelopes[right] is not None:
            total += max(0.0, min(envelopes[left][1], envelopes[right][1])
                         - max(envelopes[left][0], envelopes[right][0]))
    return total


def _first_task_order_risk(
    routes: Sequence[Sequence[str]], blocks: Mapping[str, WeldingBlock]
) -> float:
    return sum(
        max(0.0, _block_mid_x(blocks[routes[left][0]])
            - _block_mid_x(blocks[routes[right][0]]))
        for left, right in ((0, 1), (2, 3))
        if routes[left] and routes[right]
    )


def _crrr_components(
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *, current_max_load: float, current_travel: float, wait_duration: float,
) -> tuple[float, float, float, float, float, float]:
    loads = _loads(routes, blocks, config)
    travel = _directed_proxy(routes, hints, blocks, config)
    scale = max(current_max_load, 1.0)
    travel_scale = max(current_travel, 1.0)
    overlap = _route_overlap_proxy(routes, blocks) / max(config.workspace_x[1] - config.workspace_x[0], 1.0)
    order_risk = _first_task_order_risk(routes, blocks) / max(config.workspace_x[1] - config.workspace_x[0], 1.0)
    normalized = (
        0.45 * max(loads) / scale
        + 0.20 * (max(loads) - min(loads)) / scale
        + 0.20 * travel / travel_scale
        + 0.10 * overlap
        + 0.05 * order_risk * (1.0 + min(wait_duration / scale, 1.0))
    )
    return normalized, max(loads), max(loads) - min(loads), travel, overlap, order_risk


def build_crrr_candidate(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *, attempt_index: int, insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096,
) -> RepairResult:
    """Build one joint reassignment/reordering without direction or B32 access."""
    region = identify_crrr_region(current, schedule, config, attempt_index=attempt_index)
    if region is None:
        return RepairResult(None, 0, "CRRR_NO_REGION")
    selected = set(region.affected_parent_ids)
    blocks = block_map(current, config)
    patterns = {pattern.parent_id: pattern for pattern in current.patterns}
    robot_by_block = _robot_by_block(current)
    current_routes = tuple(route.block_ids for route in current.routes)
    hints_by_block = _hints_from_current(current, current_directions)
    current_hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in current_routes)
    routes = tuple(tuple(item for item in route if _parent_id(item) not in selected) for route in current_routes)
    hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in routes)
    current_loads = _loads(current_routes, blocks, config)
    current_travel = _directed_proxy(current_routes, current_hints, blocks, config)
    scheme = attempt_index % 4
    scheme_names = (
        "SAME_RAIL_CONTIGUOUS_TRANSFER", "SAME_RAIL_BOUNDARY_EXCHANGE",
        "WAIT_REGION_LOCAL_REORDER", "CROSS_RAIL_REALLOCATION",
    )
    projected_loads = list(_loads(routes, blocks, config))
    placements: list[tuple[str, int]] = []
    parent_order = sorted(selected, key=lambda parent_id: (
        abs(_parent_mid_x(parent_id, blocks) - region.anchor_x), parent_id,
    ))
    if scheme == 2:
        parent_order.reverse()
    for parent_rank, parent_id in enumerate(parent_order):
        parent_blocks = sorted(
            (block for block in blocks.values() if block.parent_id == parent_id),
            key=lambda block: (_block_mid_x(block), block.block_id),
            reverse=scheme == 2 and parent_rank % 2 == 0,
        )
        pattern = patterns[parent_id]
        for block in parent_blocks:
            original_robot = robot_by_block[block.block_id]
            eligible = [robot for robot in range(4) if robot_is_eligible(block, robot, config)]
            if pattern.kind is SplitKind.X_SPLIT:
                eligible = [original_robot]
            if scheme == 0:
                preferred = [region.partner_robot, original_robot]
            elif scheme == 1:
                swap = (region.partner_robot if original_robot == region.critical_robot
                        else region.critical_robot if original_robot == region.partner_robot
                        else original_robot)
                preferred = [swap, original_robot]
            elif scheme == 2:
                preferred = [original_robot]
            else:
                preferred = sorted(
                    (robot for robot in eligible if robot // 2 != original_robot // 2),
                    key=lambda robot: (projected_loads[robot], robot),
                ) + [original_robot]
            target = next((robot for robot in preferred if robot in eligible), None)
            if target is None:
                target = min(eligible, key=lambda robot: (projected_loads[robot], robot))
            placements.append((block.block_id, target))
            projected_loads[target] += config.process_time(block.length)
    evaluations = 0
    trace: list[tuple[str, int, int, int]] = []
    for block_id, target in placements:
        if evaluations >= max_insertion_evaluations:
            return RepairResult(None, evaluations, "CRRR_REPAIR_BUDGET_EXHAUSTED")
        alternatives = []
        for position in bounded_insertion_positions(len(routes[target]), insertion_limit):
            if evaluations >= max_insertion_evaluations:
                break
            evaluations += 1
            candidate_routes, candidate_hints = _insert_one(
                routes, hints, blocks[block_id], target, position, hints_by_block.get(block_id, 0)
            )
            components = _crrr_components(
                candidate_routes, candidate_hints, blocks, config,
                current_max_load=max(current_loads), current_travel=current_travel,
                wait_duration=region.wait_duration,
            )
            alternatives.append((components, position, candidate_routes, candidate_hints))
        if not alternatives:
            return RepairResult(None, evaluations, "CRRR_NO_LEGAL_INSERTION")
        _, position, routes, hints = min(alternatives, key=lambda row: (row[0], row[1]))
        trace.append((block_id, target, position, hints_by_block.get(block_id, 0)))
    try:
        solution = canonicalize(
            current.parents, current.patterns,
            tuple(Route(robot, routes[robot]) for robot in range(4)), config,
            revision=current.revision + 1,
            x_split_validator=(finite_x_split_validator(current.parents, config)
                if any(pattern.kind is SplitKind.X_SPLIT for pattern in current.patterns) else None),
        )
        if tuple(pattern.pattern_id for pattern in solution.patterns) != tuple(
            pattern.pattern_id for pattern in current.patterns
        ):
            raise ValueError("CRRR changed split pattern")
        solution_blocks = block_map(solution, config)
        if any(not robot_is_eligible(solution_blocks[item], route.robot_id, config)
               for route in solution.routes for item in route.block_ids):
            raise ValueError("robot eligibility failure")
    except (ValueError, KeyError, IndexError) as error:
        return RepairResult(None, evaluations, f"CRRR_CANONICAL_FAILURE:{error}")
    final_routes = tuple(route.block_ids for route in solution.routes)
    final_hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in final_routes)
    final_robot_by_block = _robot_by_block(solution)
    assignment_parent_ids = {
        parent_id for parent_id in selected
        if any(final_robot_by_block[block.block_id] != robot_by_block[block.block_id]
               for block in blocks.values() if block.parent_id == parent_id)
    }
    before_adjacency = _adjacency_by_parent(current_routes)
    after_adjacency = _adjacency_by_parent(final_routes)
    adjacency_parent_ids = {
        parent_id for parent_id in selected
        if before_adjacency.get(parent_id, set()) != after_adjacency.get(parent_id, set())
    }
    changed_parent_ids = assignment_parent_ids | adjacency_parent_ids
    if len(changed_parent_ids) < 2 or solution.canonical_hash == current.canonical_hash:
        return RepairResult(None, evaluations, "CRRR_NO_JOINT_CHANGE")
    final_loads = _loads(final_routes, solution_blocks, config)
    components = _crrr_components(
        final_routes, final_hints, solution_blocks, config,
        current_max_load=max(current_loads), current_travel=current_travel,
        wait_duration=region.wait_duration,
    )
    identity = CompleteCandidateIdentity(
        current.revision, CandidateSourceKind.LNS_REPAIRED.value,
        (CRRR_OPERATOR, region.bottleneck_type.value, attempt_index,
         region.affected_parent_ids, tuple(tuple(route) for route in final_routes)),
    )
    provenance = (
        ("operator", CRRR_OPERATOR),
        ("bottleneck_type", region.bottleneck_type.value),
        ("affected_parent_ids", region.affected_parent_ids),
        ("removed_parent_ids", region.affected_parent_ids),
        ("critical_robot", region.critical_robot),
        ("partner_robot", region.partner_robot),
        ("changed_parent_count", len(changed_parent_ids)),
        ("assignment_changes", len(assignment_parent_ids)),
        ("route_adjacency_changes", len(adjacency_parent_ids)),
        ("scheme", scheme_names[scheme]),
        ("cheap_score", tuple(round(value, 9) for value in components)),
        ("repair_trace", tuple(
            (parent_id, tuple(row for row in trace if _parent_id(row[0]) == parent_id))
            for parent_id in region.affected_parent_ids
        )),
    )
    return RepairResult(
        CompleteSearchCandidate(
            CandidateSourceKind.LNS_REPAIRED, solution, identity, max(final_loads),
            _directed_proxy(final_routes, final_hints, solution_blocks, config) - current_travel,
            0, (max(final_loads) - min(final_loads)) - (max(current_loads) - min(current_loads)),
            provenance,
        ),
        evaluations,
    )


@dataclass
class AdaptiveOperatorState:
    reaction: float = 0.2
    segment_length: int = 20
    weights: dict[tuple[DestroyOperator, RepairOperator], float] = field(
        default_factory=lambda: {pair: 1.0 for pair in OPERATOR_PAIRS}
    )
    segment_uses: dict[tuple[DestroyOperator, RepairOperator], int] = field(
        default_factory=lambda: {pair: 0 for pair in OPERATOR_PAIRS}
    )
    segment_rewards: dict[tuple[DestroyOperator, RepairOperator], float] = field(
        default_factory=lambda: {pair: 0.0 for pair in OPERATOR_PAIRS}
    )
    observations: int = 0
    selection_history: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.reaction <= 1.0):
            raise ValueError("adaptive reaction must be in (0,1]")
        if self.segment_length < 1:
            raise ValueError("adaptive segment length must be positive")

    def select(self, rng: random.Random) -> tuple[DestroyOperator, RepairOperator]:
        pairs = OPERATOR_PAIRS
        total = sum(self.weights[pair] for pair in pairs)
        draw = rng.random() * total
        cumulative = 0.0
        selected = pairs[-1]
        for pair in pairs:
            cumulative += self.weights[pair]
            if draw < cumulative:
                selected = pair
                break
        self.selection_history.append((selected[0].value, selected[1].value))
        return selected

    def record(
        self,
        pair: tuple[DestroyOperator, RepairOperator],
        reward: float | None,
    ) -> None:
        if reward is None:
            return
        self.segment_uses[pair] += 1
        self.segment_rewards[pair] += reward
        self.observations += 1
        if self.observations % self.segment_length:
            return
        for operator_pair in OPERATOR_PAIRS:
            uses = self.segment_uses[operator_pair]
            if uses:
                average = self.segment_rewards[operator_pair] / uses
                self.weights[operator_pair] = (
                    (1.0 - self.reaction) * self.weights[operator_pair]
                    + self.reaction * average
                )
            self.segment_uses[operator_pair] = 0
            self.segment_rewards[operator_pair] = 0.0


def lns_pair(candidate: CompleteSearchCandidate) -> tuple[DestroyOperator, RepairOperator] | None:
    if candidate.source_kind is not CandidateSourceKind.LNS_REPAIRED:
        return None
    values = dict(candidate.provenance)
    if values.get("operator") == CRRR_OPERATOR:
        return None
    if "destroy_operator" not in values or "repair_operator" not in values:
        return None
    return (
        DestroyOperator(str(values["destroy_operator"])),
        RepairOperator(str(values["repair_operator"])),
    )
