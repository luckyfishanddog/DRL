"""Load-aware four-robot adjacency-list EAX with constrained NSGA-II.

This is the explicitly documented line-weld adaptation of Gao et al. section
4.4 and Fig. 9, not AB-cycle EAX or an exact clone of its two-robot point model.
Robot ownership is decided anew at every directed insertion. No local search,
route direction postprocessing or NATIVE call occurs during evolution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import numpy as np
from numba import njit

from mrta_baselines.hga import _HGAWorkspace, _numeric_edges, initialize_hga_region_seed
from mrta_baselines.wag_vns import _priority_arrays
from mrta_reference.model import OperationKind, ScientificConfig
from mrta_reference.scheduler import native_schedule
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_search.saeax import Individual, SAEAXResult, complete_identity, validate_individual, _route_statistics


@dataclass(frozen=True)
class EAXNSGAConfig:
    population_size: int = 20
    offspring_per_generation: int = 10
    max_generations: int = 500
    min_generations: int = 100
    stagnation_generations: int = 120
    retries_per_child: int = 12
    pattern_mutation_probability: float = .15
    adjacency_mutation_probability: float = .10

    def __post_init__(self):
        for name in ('population_size','offspring_per_generation','max_generations',
                     'min_generations','stagnation_generations','retries_per_child'):
            value = getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<=0:
                raise ValueError(f'{name} must be a positive integer')
        if self.population_size<2 or self.min_generations>self.max_generations:
            raise ValueError('invalid population or generation bounds')
        for name in ('pattern_mutation_probability','adjacency_mutation_probability'):
            if not 0.<=getattr(self,name)<=1.:
                raise ValueError(f'{name} must be in [0,1]')


def enhanced_adjacency(first, second, selected, workspace, rng, settings):
    """Filter parental predecessor/successor pairs, then mutate their union.

    The fixed adaptation uses Q75 of actual parental directed MOVE distances.
    Gao section 4.4 keeps both neighbours when both exceed the threshold;
    with one short and one long, it keeps the short one. The text omits the
    both-short and open-route endpoint cases: keep both short neighbours or
    the sole endpoint neighbour. No unrelated near edges are manufactured.
    Numeric active-node mapping excludes alternative geometries with the same
    printable child ID. No parent robot index is stored in the table.
    """
    active = np.array(sorted(int(n) for p in selected for n in workspace.pattern_nodes[p] if n>=0),np.int64)
    local = np.full(len(workspace.ids),-1,np.int64)
    local[active] = np.arange(len(active))
    table = np.zeros((2*len(active),2*len(active)),np.bool_)
    inherited = np.zeros_like(table)
    distances,neighbours = [],[]
    for parent in (first,second):
        for route,bits in zip(parent.routes,parent.directions):
            for k in range(1,len(route)):
                u,v,du,dv = int(route[k-1]),int(route[k]),int(bits[k-1]),int(bits[k])
                if local[u]<0 or local[v]<0:
                    continue
                inherited[2*local[u]+du,2*local[v]+dv] = True
                inherited[2*local[v]+1-dv,2*local[u]+1-du] = True
                distances.append(workspace.travel[u,v,du,dv])
            for k,node in enumerate(route):
                node,d = int(node),int(bits[k])
                if local[node]<0:continue
                pair = []
                if k and local[route[k-1]]>=0:
                    previous,dp = int(route[k-1]),int(bits[k-1])
                    pair.append((2*local[node]+1-d,2*local[previous]+1-dp,
                        float(workspace.travel[previous,node,dp,d])))
                if k+1<len(route) and local[route[k+1]]>=0:
                    following,df = int(route[k+1]),int(bits[k+1])
                    pair.append((2*local[node]+d,2*local[following]+df,
                        float(workspace.travel[node,following,d,df])))
                if pair:neighbours.append(pair)
    threshold = float(np.quantile(distances,.75)) if distances else 0.
    for pair in neighbours:
        both_long = all(distance>threshold for _,_,distance in pair)
        for a,b,distance in pair:
            if len(pair)==1 or both_long or distance<=threshold:table[a,b] = True
    threshold_entries = int(table.sum())
    mutations = 0
    # Insert the sampled node into another node's adjacency list, including
    # the geometrically equivalent reverse-complement directed transition.
    for i,u in enumerate(active):
        if rng.random()>=settings.adjacency_mutation_probability:
            continue
        targets = [j for j,v in enumerate(active) if workspace.parent_index[v]!=workspace.parent_index[u]]
        if not targets:
            continue
        j = rng.choice(targets)
        v = active[j]
        du,dv = min(((a,b) for a in range(2) for b in range(2)),
                    key=lambda ab:workspace.travel[v,u,ab[0],ab[1]])
        for a,b in ((2*j+du,2*i+dv),(2*i+1-dv,2*j+1-du)):
            if not table[a,b]:
                table[a,b] = True
                mutations += 1
    return active,table,inherited,dict(distance_threshold_m=threshold*workspace.config.empty_speed,
        parental_adjacency_entries=int(inherited.sum()),threshold_adjacency_entries=threshold_entries,
        threshold_excluded_entries=int(inherited.sum())-threshold_entries,
        mutation_entries=mutations)


@njit
def _joint_position(node, preferred, local, table, routes, bits, lengths, loads, costs,
                    travel, processing, eligible, parent_index, points, parameters):
    answer = (-1,0,0,0.)
    best = (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)
    ties = np.random.random(4)
    for r in range(4):
        if not eligible[node,r]:
            continue
        other_p,other_cycle = 0.,0.
        for s in range(4):
            if s!=r:
                other_p,other_cycle = max(other_p,loads[s]),max(other_cycle,costs[s])
        peak_p = max(loads[r]+processing[node],other_p)
        for k in range(lengths[r]+1):
            if k and parent_index[routes[r,k-1]]==parent_index[node]:
                continue
            if k<lengths[r] and parent_index[routes[r,k]]==parent_index[node]:
                continue
            for d in range(2):
                if k==0:
                    a = points[node,d]
                    legal = True
                    for s in range(4):
                        if s==r or not lengths[s]:
                            continue
                        b = points[routes[s,0],bits[s,0]]
                        if r//2==s//2 and ((r<s and a[0]+parameters[3]>b[0]+parameters[5]) or
                                (s<r and b[0]+parameters[3]>a[0]+parameters[5])):
                            legal = False
                        if abs(a[0]-b[0])<=parameters[3]+parameters[5] and abs(a[1]-b[1])<=parameters[4]+parameters[5]:
                            legal = False
                    if not legal:
                        continue
                delta, support = 0.,0
                if k:
                    u,du = routes[r,k-1],bits[r,k-1]
                    delta += travel[u,node,du,d]
                    support += int(table[2*local[u]+du,2*local[node]+d])
                if k<lengths[r]:
                    v,dv = routes[r,k],bits[r,k]
                    delta += travel[node,v,d,dv]
                    support += int(table[2*local[node]+d,2*local[v]+dv])
                if k and k<lengths[r]:
                    delta -= travel[routes[r,k-1],routes[r,k],bits[r,k-1],bits[r,k]]
                cycle = max(other_cycle,costs[r]+processing[node]+delta)
                key = (peak_p,cycle,float(-support),delta,1. if d!=preferred else 0.,ties[r],float(k))
                if key<best:
                    best,answer = key,(r,k,d,delta)
    return answer


@njit
def _construct_joint(active, table, travel, processing, eligible, parent_index, points, parameters, seed):
    np.random.seed(seed)
    n = len(active)
    routes,bits = np.zeros((4,n),np.int64),np.zeros((4,n),np.int64)
    lengths,loads,costs = np.zeros(4,np.int64),np.zeros(4),np.zeros(4)
    local = np.full(len(processing),-1,np.int64)
    for i,node in enumerate(active):local[node] = i
    degree = np.sum(table,axis=1)
    used = np.zeros(n,np.bool_)
    last = -1
    # actual few-neighbour choices, nearest fallbacks, deferred illegal nodes
    counts = np.zeros(3,np.int64)
    for step in range(n):
        candidates = []
        for i,node in enumerate(active):
            if used[i]:continue
            preferred,adjacent,remaining,distance = 0,False,np.inf,np.inf
            for d in range(2):
                near = last>=0 and table[last,2*i+d]
                dist = travel[active[last//2],node,last%2,d] if last>=0 else 0.
                key = (not near,float(degree[2*i+d]) if near else dist,dist,d)
                old = (not adjacent,remaining,distance,preferred)
                if key<old:
                    adjacent,remaining,distance,preferred = near,key[1],dist,d
            # Initial node is seeded random. Subsequently the canonical
            # fewest-remaining-neighbours rule leads, nearest MOVE otherwise.
            priority = remaining if last>=0 else 0.
            candidates.append((int(not adjacent),priority,distance,np.random.random(),i,preferred))
        candidates.sort()
        placed = False
        for _,_,_,_,i,preferred in candidates:
            node = active[i]
            r,k,d,delta = _joint_position(node,preferred,local,table,routes,bits,lengths,loads,costs,
                travel,processing,eligible,parent_index,points,parameters)
            if r<0:
                counts[2] += 1
                continue
            if last>=0:
                if table[last,2*i] or table[last,2*i+1]:counts[0] += 1
                else:counts[1] += 1
            for j in range(lengths[r],k,-1):
                routes[r,j],bits[r,j] = routes[r,j-1],bits[r,j-1]
            routes[r,k],bits[r,k] = node,d
            lengths[r] += 1
            loads[r] += processing[node]
            costs[r] += processing[node]+delta
            used[i] = True
            for j in range(len(degree)):
                degree[j] -= int(table[j,2*i])+int(table[j,2*i+1])
            last = 2*i+d
            placed = True
            break
        if not placed:
            return routes,bits,lengths,counts,False
    return routes,bits,lengths,counts,True


def adjacency_crossover(first, second, workspace, rng, settings):
    selected = np.array([a if rng.random()<.5 else b for a,b in zip(first.selected,second.selected)],np.int64)
    if rng.random()<settings.pattern_mutation_probability:
        mutable = [p for p,options in enumerate(workspace.options) if len(options)>1]
        if mutable:
            p = rng.choice(mutable)
            selected[p] = rng.choice([v for v in workspace.options[p] if v!=selected[p]])
    active,table,inherited,detail = enhanced_adjacency(first,second,selected,workspace,rng,settings)
    rr,dd,lengths,counts,complete = _construct_joint(active,table,workspace.travel,workspace.processing,
        workspace.eligible,workspace.parent_index,workspace.points,workspace.parameters,rng.randrange(2**32))
    if not complete:
        raise ValueError('no legal joint insertion for remaining active nodes')
    routes = [rr[r,:lengths[r]].copy() for r in range(4)]
    directions = [dd[r,:lengths[r]].copy() for r in range(4)]
    validate_individual(workspace,routes,directions,selected)
    local = {int(n):i for i,n in enumerate(active)}
    kept = sum(int(inherited[2*local[int(row[k-1])]+int(bits[k-1]),2*local[int(row[k])]+int(bits[k])])
        for row,bits in zip(routes,directions) for k in range(1,len(row)))
    detail.update(few_neighbour_choices=int(counts[0]),nearest_fallbacks=int(counts[1]),
        deferred_nodes=int(counts[2]),inherited_edges=kept,
        new_edges=sum(max(0,len(row)-1) for row in routes)-kept)
    for label,parent in (('first',first),('second',second)):
        owners = {int(n):r for r,row in enumerate(parent.routes) for n in row}
        detail['owner_changes_vs_'+label] = sum(int(n) in owners and owners[int(n)]!=r
            for r,row in enumerate(routes) for n in row)
    return routes,directions,selected,detail


def objectives(individual):
    return (float(individual.native_cmax),float(max(individual.processing)),
            float(sum(individual.costs-individual.processing)))


def final_key(individual):
    return ((False,*objectives(individual),individual.identity) if math.isfinite(individual.native_cmax)
            else (True,math.inf,math.inf,math.inf,individual.identity))


@njit
def rank_and_crowding(values):
    """Deb constraint dominance, nondominated fronts and normalized crowding.

    The native evaluator reports feasibility, not a numeric violation amount.
    Rejected individuals are equivalent under constraint dominance; their
    finite P/M values are never substituted for a feasible native objective.
    """
    n = len(values)
    dominates = np.zeros((n,n),np.bool_)
    dominated_by = np.zeros(n,np.int64)
    feasible = np.isfinite(values[:,0])
    for i in range(n):
        for j in range(i+1,n):
            a,b = False,False
            if feasible[i]!=feasible[j]:a,b = feasible[i],feasible[j]
            elif feasible[i]:
                a = np.all(values[i]<=values[j]) and np.any(values[i]<values[j])
                b = np.all(values[j]<=values[i]) and np.any(values[j]<values[i])
            if a:dominates[i,j],dominated_by[j] = True,dominated_by[j]+1
            if b:dominates[j,i],dominated_by[i] = True,dominated_by[i]+1
    rank,crowding = np.full(n,-1,np.int64),np.zeros(n)
    front_number = 0
    while np.any(rank<0):
        front = np.where((dominated_by==0)&(rank<0))[0]
        for i in front:rank[i] = front_number
        if feasible[front[0]]:
            if len(front)<=2:
                for i in front:crowding[i] = np.inf
            else:
                for axis in range(3):
                    order = front[np.argsort(values[front,axis],kind='mergesort')]
                    span = values[order[-1],axis]-values[order[0],axis]
                    if span<=0:continue
                    crowding[order[0]],crowding[order[-1]] = np.inf,np.inf
                    for k in range(1,len(order)-1):
                        crowding[order[k]] += (values[order[k+1],axis]-values[order[k-1],axis])/span
        for i in front:
            for j in range(n):
                if dominates[i,j]:dominated_by[j] -= 1
        front_number += 1
    return rank,crowding


def select_population(population, size):
    population = list({item.identity:item for item in population}.values())
    values = np.array([objectives(item) for item in population])
    ranks,crowding = rank_and_crowding(values)
    order = sorted(range(len(population)),key=lambda i:(int(ranks[i]),-float(crowding[i]),final_key(population[i])))
    return [population[i] for i in order[:size]]


def _tournament(population, ranks, crowding, rng):
    a,b = rng.sample(range(len(population)),2)
    ka,kb = (ranks[a],-crowding[a]),(ranks[b],-crowding[b])
    return population[a if ka<kb or (ka==kb and rng.random()<.5) else b]


def run_eax_nsga2(parents, config=None, settings=None, *, seed=20261110,
                  scope=FORMAL_SCOPE_V2, progress_callback=None):
    config,settings = config or ScientificConfig(),settings or EAXNSGAConfig()
    started,rng = time.perf_counter(),random.Random(seed)
    w = _HGAWorkspace(parents,config,scope)
    accounting = dict(native_calls=0,coordination_calls=0,initialization_evaluations=0,
        initialization_attempts=0,construction_attempts=0,duplicate_children=0,invalid_children=0,
        native_feasible_children=0,crossover_s=0.,native_s=0.,population_s=0.,coordination_s=0.,
        best_improvements=0,inherited_edges=0,new_edges=0,mutation_entries=0,
        threshold_adjacency_entries=0,threshold_excluded_entries=0,parental_adjacency_entries=0,few_neighbour_choices=0,
        nearest_fallbacks=0,deferred_nodes=0,owner_changes_vs_first=0,owner_changes_vs_second=0,
        distance_threshold_m_sum=0.,evaluated_eax_children=0)
    seen,population,progress = {},[],[]

    def evaluate(routes,bits,selected,detail):
        validate_individual(w,routes,bits,selected)
        identity = complete_identity(routes,bits,selected)
        if identity in seen:
            accounting['duplicate_children'] += 1
            return None
        t = time.perf_counter()
        native,_,_,_ = _priority_arrays(routes,w.travel,w.processing,w.points,w.parameters,bits)
        accounting['native_calls'] += 1
        accounting['native_s'] += time.perf_counter()-t
        if math.isnan(native):raise ArithmeticError('native scorer returned NaN')
        costs,loads = _route_statistics(routes,bits,w.travel,w.processing)
        edges = _numeric_edges([2*r+d for r,d in zip(routes,bits)],2*len(w.ids))
        item = Individual(routes,bits,selected,float(native),costs,loads,identity,edges,
                          detail['inherited_edges'],detail['new_edges'])
        seen[identity] = item
        accounting['evaluated_eax_children'] += 1
        accounting['distance_threshold_m_sum'] += detail['distance_threshold_m']
        for key in ('inherited_edges','new_edges','mutation_entries','threshold_adjacency_entries','threshold_excluded_entries',
                    'parental_adjacency_entries','few_neighbour_choices','nearest_fallbacks',
                    'deferred_nodes','owner_changes_vs_first','owner_changes_vs_second'):
            accounting[key] += detail[key]
        return item

    def child(first,second,initializing=False):
        for _ in range(settings.retries_per_child):
            accounting['initialization_attempts' if initializing else 'construction_attempts'] += 1
            t = time.perf_counter()
            try:
                r,d,s,detail = adjacency_crossover(first,second,w,rng,settings)
            except ValueError:
                accounting['crossover_s'] += time.perf_counter()-t
                accounting['invalid_children'] += 1
                continue
            accounting['crossover_s'] += time.perf_counter()-t
            item = evaluate(r,d,s,detail)
            if item is not None:return item
        return None

    # The old unsearched REGION solution supplies adjacency only, not fixed
    # offspring owners or an already evaluated baseline solution. Every member
    # of the new initial population is jointly constructed by the new operator.
    reference = initialize_hga_region_seed(parents,config,scope=scope)
    selected = np.array([w.patterns.index(p) for p in reference.patterns],np.int64)
    routes = w.encode(reference)
    bits = [np.zeros(len(r),np.int64) for r in routes]
    costs,loads = _route_statistics(routes,bits,w.travel,w.processing)
    template = Individual(routes,bits,selected,math.inf,costs,loads,
                          complete_identity(routes,bits,selected),np.empty(0,np.int64))
    initialization_failed = False
    for _ in range(settings.population_size):
        item = child(template,rng.choice(population) if population else template,True)
        if item is None:
            initialization_failed = True
            break
        population.append(item)
    accounting['initialization_evaluations'] = len(seen)
    accounting['initialization_s'] = time.perf_counter()-started
    if not population:
        accounting.update(search_complete=False,search_s=time.perf_counter()-started,
            population_size_at_stop=0,stagnation_at_stop=0,partial_generation_children=0)
        return SAEAXResult('CONSTRUCTION_FAILURE',None,None,None,None,None,0,0,
            'INITIALIZATION_RETRIES_EXHAUSTED',time.perf_counter()-started,accounting,[],[])
    best = min(population,key=final_key)
    accounting['initial_native_cmax_s'] = best.native_cmax if math.isfinite(best.native_cmax) else None
    accounting['initial_max_processing_s'] = float(max(best.processing))
    generations,children,stale,partial_children = 0,0,0,0
    termination = 'INITIALIZATION_RETRIES_EXHAUSTED' if initialization_failed else 'MAX_GENERATIONS'

    def snapshot():
        ranks,_ = rank_and_crowding(np.array([objectives(item) for item in population]))
        row = dict(generation=generations,valid_children=children,native_calls=accounting['native_calls'],
            elapsed_s=time.perf_counter()-started,native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
            max_processing_s=float(max(best.processing)),move_s=float(sum(best.costs-best.processing)),
            feasible_population=sum(math.isfinite(p.native_cmax) for p in population),
            pareto_population=int(sum(ranks[i]==0 and math.isfinite(p.native_cmax)
                for i,p in enumerate(population))),stagnation=stale)
        progress.append(row)
        if progress_callback is not None:progress_callback(row)

    snapshot()
    while not initialization_failed and generations<settings.max_generations:
        old_best,batch = best.native_cmax,[]
        t = time.perf_counter()
        ranks,crowding = rank_and_crowding(np.array([objectives(p) for p in population]))
        accounting['population_s'] += time.perf_counter()-t
        for _ in range(settings.offspring_per_generation):
            first,second = (_tournament(population,ranks,crowding,rng) for _ in range(2))
            item = child(first,second)
            if item is None:break
            batch.append(item)
            children += 1
            accounting['native_feasible_children'] += int(math.isfinite(item.native_cmax))
            if final_key(item)<final_key(best):
                accounting['best_improvements'] += int(item.native_cmax<best.native_cmax-1.e-8)
                best = item
        t = time.perf_counter()
        population = select_population(population+batch,settings.population_size)
        accounting['population_s'] += time.perf_counter()-t
        if len(batch)!=settings.offspring_per_generation:
            termination,partial_children = 'NOVEL_CHILD_RETRIES_EXHAUSTED',len(batch)
            break
        generations += 1
        stale = 0 if best.native_cmax<old_best-1.e-8 else stale+1
        if generations%25==0:snapshot()
        if generations>=settings.min_generations and stale>=settings.stagnation_generations:
            termination = 'STAGNATION'
            break
    if progress[-1]['generation']!=generations or partial_children:snapshot()
    accounting.update(stagnation_at_stop=stale,partial_generation_children=partial_children,
        population_size_at_stop=len(population),
        search_complete=termination in ('STAGNATION','MAX_GENERATIONS'))
    # Lexicographic minimum of feasible objective vectors is necessarily a
    # nondominated point of ALL evaluated solutions. Reuse the ordinary seen
    # dictionary, without a second archive or more evaluations. This also keeps
    # the global native extreme if crowding discarded its duplicate objective.
    final = min(seen.values(),key=final_key)
    solution = w.solution(final.routes,final.selected)
    expected = tuple(tuple(w.ids[n] for n in row) for row in final.routes)
    if tuple(r.block_ids for r in solution.routes)!=expected or solution.patterns!=tuple(w.patterns[p] for p in final.selected):
        raise RuntimeError('canonicalization changed the final directed individual')
    vectors = tuple(tuple(int(d) for d in row) for row in final.directions)
    accounting['search_s'] = time.perf_counter()-started
    t = time.perf_counter()
    accounting['coordination_calls'] = 1
    schedule = native_schedule(solution,config,scope=scope,orientations=dict(enumerate(vectors)))
    accounting['coordination_s'] = time.perf_counter()-t
    if schedule.feasible and schedule.directions!=vectors:
        raise RuntimeError('NATIVE changed explicit directions')
    native = final.native_cmax if math.isfinite(final.native_cmax) else None
    accounting.update(native_best_cmax_s=native,processing_s=[float(p) for p in final.processing],
        move_s=float(sum(final.costs-final.processing)),
        wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
        wait_is_partial=not schedule.feasible,
        final_selection='NATIVE_PMAX_MOVE_LEXICOGRAPHIC_NONDOMINATED_POINT')
    final_rows = [dict(rank=1,native_cmax_s=native,status=schedule.status.value,
        cmax_s=schedule.cmax if schedule.feasible else None,coordination_s=accounting['coordination_s'],
        max_processing_s=float(max(final.processing)),move_s=accounting['move_s'],wait_s=accounting['wait_s'],
        diagnostic=' | '.join(schedule.diagnostics[:2]))]
    return SAEAXResult(schedule.status.value,schedule.cmax if schedule.feasible else None,
        solution,vectors,schedule,native,generations,children,termination,time.perf_counter()-started,
        accounting,progress,final_rows)
