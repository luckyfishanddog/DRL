"""Four-robot scheduling-aware enhanced adjacency crossover (4R-SAEAX).

This is a project-specific adjacency-list method, not a reproduction of the
unavailable full operator in Gao et al. (2026), nor TSP alternating-cycle EAX.
Geometry, qualifications and both scheduling models belong to the existing
reference implementation. No route VND or direction-optimization phase is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import numpy as np
from numba import njit

from mrta_baselines.hga import (
    _HGAWorkspace, _greedy_route_directions, _numeric_edges,
    _population_diversity, _repair_oriented_first, initialize_hga_region_seed,
)
from mrta_baselines.wag_vns import _priority_arrays
from mrta_reference.model import OperationKind, ScheduleStatus, ScientificConfig
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import FORMAL_SCOPE_V2


@dataclass(frozen=True)
class SAEAXConfig:
    population_size: int = 12
    offspring_per_generation: int = 4
    max_generations: int = 400
    min_generations: int = 80
    stagnation_generations: int = 120
    retries_per_child: int = 12
    pattern_mutation_probability: float = 0.15
    relocate_probability: float = 0.25
    swap_probability: float = 0.10
    max_fragment_length: int = 6

    def __post_init__(self):
        for name in ('population_size', 'offspring_per_generation', 'max_generations',
                     'min_generations', 'stagnation_generations', 'retries_per_child',
                     'max_fragment_length'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.population_size < 2 or self.min_generations > self.max_generations:
            raise ValueError('invalid population or generation bounds')
        for name in ('pattern_mutation_probability', 'relocate_probability', 'swap_probability'):
            if not 0. <= getattr(self, name) <= 1.:
                raise ValueError(f'{name} must be in [0,1]')


@dataclass
class Individual:
    routes: list
    directions: list
    selected: np.ndarray
    native_cmax: float
    costs: np.ndarray
    processing: np.ndarray
    identity: tuple
    edges: np.ndarray
    inherited_edges: int = 0
    new_edges: int = 0

    @property
    def key(self):
        return (not math.isfinite(self.native_cmax), self.native_cmax,
                float(max(self.costs)), float(sum(self.costs)), self.identity)


@dataclass
class SAEAXResult:
    status: str
    cmax: float | None
    solution: object | None
    directions: tuple | None
    schedule: object | None
    native_cmax: float | None
    generations: int
    valid_children: int
    termination_reason: str
    runtime: float
    accounting: dict
    progress: list = field(default_factory=list)
    final_candidates: list = field(default_factory=list)


@njit
def _route_statistics(routes, directions, travel, processing):
    loads, costs = np.zeros(4), np.zeros(4)
    for r in range(4):
        for k, node in enumerate(routes[r]):
            loads[r] += processing[node]
            if k:
                costs[r] += travel[routes[r][k-1], node, directions[r][k-1], directions[r][k]]
        costs[r] += loads[r]
    return costs, loads


@njit
def _heads_legal(routes, directions, points, parameters):
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for r in range(4):
        if not len(routes[r]):
            continue
        a = points[routes[r][0], directions[r][0]]
        for s in range(r+1, 4):
            if not len(routes[s]):
                continue
            b = points[routes[s][0], directions[s][0]]
            if r//2 == s//2 and a[0] + dx > b[0] + eps:
                return False
            if abs(a[0]-b[0]) <= dx+eps and abs(a[1]-b[1]) <= dy+eps:
                return False
    return True


@njit
def _fragment_position(fragment, bits, routes, directions, remaining, costs,
                       preferred, travel, processing, eligible, parent_index,
                       points, parameters):
    """Joint robot/position/orientation insertion; no direction postprocessing.

    remaining contains the processing still reserved for other fragments at
    their inherited owners. Thus a partially built short route cannot attract
    all flexible work simply because its later inherited work is not inserted.
    """
    best, best_delta, best_owner = np.inf, np.inf, 3
    answer = (-1, 0, 0)
    load = 0.
    for n in fragment:
        load += processing[n]
    for r in range(4):
        qualified = True
        for n in fragment:
            if not eligible[n,r]:
                qualified = False
                break
        if not qualified:
            continue
        other = 0.
        for s in range(4):
            if s != r:
                other = max(other,costs[s]+remaining[s])
        for reverse in range(2):
            nodes = fragment if reverse == 0 else fragment[::-1]
            vector = bits if reverse == 0 else 1-bits[::-1]
            internal = load
            for j in range(1, len(nodes)):
                internal += travel[nodes[j-1], nodes[j], vector[j-1], vector[j]]
            for k in range(len(routes[r])+1):
                if k and parent_index[routes[r][k-1]] == parent_index[nodes[0]]:
                    continue
                if k < len(routes[r]) and parent_index[routes[r][k]] == parent_index[nodes[-1]]:
                    continue
                delta = internal
                if k:
                    delta += travel[routes[r][k-1], nodes[0], directions[r][k-1], vector[0]]
                if k < len(routes[r]):
                    delta += travel[nodes[-1], routes[r][k], vector[-1], directions[r][k]]
                if k and k < len(routes[r]):
                    delta -= travel[routes[r][k-1], routes[r][k], directions[r][k-1], directions[r][k]]
                if k == 0:
                    # The same initial point constraints as the existing native
                    # evaluator, applied while choosing this insertion's state.
                    a = points[nodes[0], vector[0]]
                    legal = True
                    for s in range(4):
                        if s == r or not len(routes[s]):
                            continue
                        b = points[routes[s][0], directions[s][0]]
                        if r//2 == s//2:
                            if ((r < s and a[0]+parameters[3] > b[0]+parameters[5]) or
                                (s < r and b[0]+parameters[3] > a[0]+parameters[5])):
                                legal = False
                        if (abs(a[0]-b[0]) <= parameters[3]+parameters[5] and
                            abs(a[1]-b[1]) <= parameters[4]+parameters[5]):
                            legal = False
                    if not legal:
                        continue
                peak = max(costs[r]+delta+remaining[r], other)
                owner = int(r != preferred)
                if (peak < best-1.e-9 or (abs(peak-best) <= 1.e-9 and
                    (delta < best_delta-1.e-9 or (abs(delta-best_delta) <= 1.e-9 and owner < best_owner)))):
                    best, best_delta, best_owner = peak, delta, owner
                    answer = (r, k, reverse)
    return answer


def _put_fragment(fragment, bits, routes, directions, position):
    robot, index, reverse = position
    if robot < 0:
        raise ValueError('no legal directed fragment insertion')
    if reverse:
        fragment, bits = fragment[::-1], 1-bits[::-1]
    routes[robot] = np.concatenate((routes[robot][:index], fragment, routes[robot][index:]))
    directions[robot] = np.concatenate((directions[robot][:index], bits, directions[robot][index:]))


def validate_individual(workspace, routes, directions, selected):
    """Cheap numeric integrity checks before native scoring; canonicalize at export."""
    if len(routes) != 4 or len(directions) != 4 or len(selected) != len(workspace.parents):
        raise ValueError('invalid four-robot encoding')
    if any(int(p) not in workspace.options[i] for i, p in enumerate(selected)):
        raise ValueError('illegal parent mode')
    expected = sorted(int(n) for p in selected for n in workspace.pattern_nodes[p] if n >= 0)
    actual = [int(n) for route in routes for n in route]
    if sorted(actual) != expected:
        raise ValueError('missing, duplicate or inactive child node')
    for r, (route, bits) in enumerate(zip(routes, directions)):
        if len(route) != len(bits) or any(d not in (0, 1) for d in bits):
            raise ValueError('direction and node are not aligned')
        if any(not workspace.eligible[n, r] for n in route):
            raise ValueError('robot qualification or fixed X ownership violated')
        if any(workspace.parent_index[a] == workspace.parent_index[b] for a, b in zip(route, route[1:])):
            raise ValueError('adjacent split siblings would be canonicalized to WHOLE')


def complete_identity(routes, directions, selected):
    # Ordinary immutable tuple keys, including alternative split geometry; no
    # hash protocol or persistent identity system is introduced.
    return (tuple(int(p) for p in selected), tuple(
        tuple((int(n), int(d)) for n, d in zip(route, bits))
        for route, bits in zip(routes, directions)))


def enhanced_adjacency(first, second, selected, workspace):
    """Each directed state has successor and reversed-predecessor entries.

    Entries retain robot, parental native quality, local MOVE and parent index.
    An alternative geometry with the same printable child ID is never mapped
    onto the selected geometry: only exactly active numeric nodes survive.
    """
    active = {int(n) for p in selected for n in workspace.pattern_nodes[p] if n >= 0}
    table = {}
    owners = {n: [] for n in active}
    for parent_id, parent in enumerate((first, second)):
        for r, (route, bits) in enumerate(zip(parent.routes, parent.directions)):
            for node in route:
                if int(node) in active:
                    owners[int(node)].append((r, parent_id))
            for k in range(1, len(route)):
                u, v, du, dv = int(route[k-1]), int(route[k]), int(bits[k-1]), int(bits[k])
                if u not in active or v not in active:
                    continue
                quality = (not math.isfinite(parent.native_cmax), parent.native_cmax,
                           float(parent.costs[r]), float(workspace.travel[u,v,du,dv]))
                for a, da, b, db in ((u,du,v,dv), (v,1-dv,u,1-du)):
                    table.setdefault((a,da), []).append((b,db,r,quality,parent_id))
    return table, owners


def adjacency_fragments(first, second, selected, workspace, rng, settings):
    table, owners = enhanced_adjacency(first, second, selected, workspace)
    active = sorted(owners)
    arcs = {}
    for (u, du), values in table.items():
        for v, dv, robot, quality, parent in values:
            arcs.setdefault((u,du,v,dv), []).append((robot,quality,parent))
    ranked = []
    for arc, entries in arcs.items():
        support = len({item[2] for item in entries})
        # Drop some unique arcs to expose recombination boundaries. Shared
        # arcs lead, followed by arcs from better native solutions/routes.
        if support == 1 and rng.random() < .20:
            continue
        quality = min(item[1] for item in entries)
        # Close-quality parents must both contribute unique arcs. A strict
        # lexicographic parent-Cmax ordering would consume all of the better
        # parent's arcs before visiting any unique arc from its mate.
        value = quality[1] if math.isfinite(quality[1]) else quality[2]
        ranked.append((-support,quality[0],value*(.9+.2*rng.random()),quality[3],arc))
    ranked.sort()
    fragments = {n: ([n], [None]) for n in active}
    membership = {n: n for n in active}
    for _, _, _, _, (u,du,v,dv) in ranked:
        a, b = membership[u], membership[v]
        if a == b:
            continue
        left, ld = fragments[a]
        right, rd = fragments[b]
        if left[-1] != u or right[0] != v or len(left)+len(right) > settings.max_fragment_length:
            continue
        if (ld[-1] is not None and ld[-1] != du) or (rd[0] is not None and rd[0] != dv):
            continue
        if workspace.parent_index[u] == workspace.parent_index[v]:
            continue
        combined = left+right
        if not any(all(workspace.eligible[n,r] for n in combined) for r in range(4)):
            continue
        ld[-1], rd[0] = du, dv
        fragments[a] = combined, ld+rd
        del fragments[b]
        for n in right:
            membership[n] = a
    result = []
    for nodes, bits in fragments.values():
        votes = [r for n in nodes for r, _ in owners[n]
                 if all(workspace.eligible[k,r] for k in nodes)]
        legal = [r for r in range(4) if all(workspace.eligible[n,r] for n in nodes)]
        preferred = rng.choice(votes or legal)
        result.append((np.array(nodes,np.int64), np.array([0 if d is None else d for d in bits],np.int64), preferred))
    return result, arcs


def assemble_fragments(fragments, workspace, rng):
    remaining = np.zeros(4)
    for nodes, _, owner in fragments:
        remaining[owner] += sum(workspace.processing[n] for n in nodes)
    # Process constrained and heavy fragments first; random ties give different
    # recombinations without a local-search pass or repeated complete scoring.
    ordered = sorted(fragments, key=lambda f: (
        sum(all(workspace.eligible[n,r] for n in f[0]) for r in range(4)),
        -sum(workspace.processing[n] for n in f[0])*(.9+.2*rng.random())))
    routes = [np.empty(0,np.int64) for _ in range(4)]
    directions = [np.empty(0,np.int64) for _ in range(4)]
    for nodes, bits, owner in ordered:
        remaining[owner] -= sum(workspace.processing[n] for n in nodes)
        costs, _ = _route_statistics(routes,directions,workspace.travel,workspace.processing)
        position = _fragment_position(nodes,bits,routes,directions,remaining,costs,owner,
            workspace.travel,workspace.processing,workspace.eligible,workspace.parent_index,
            workspace.points,workspace.parameters)
        _put_fragment(nodes,bits,routes,directions,position)
    return routes,directions


def mutate_child(routes, directions, workspace, rng, settings):
    routes, directions = [r.copy() for r in routes], [d.copy() for d in directions]
    original = ([r.copy() for r in routes], [d.copy() for d in directions])
    costs, _ = _route_statistics(routes,directions,workspace.travel,workspace.processing)
    heavy = max(range(4),key=lambda r: costs[r])
    if not len(routes[heavy]):
        return routes,directions
    i = rng.randrange(len(routes[heavy]))
    node, bit = int(routes[heavy][i]), int(directions[heavy][i])
    if rng.random() < settings.relocate_probability:
        routes[heavy] = np.delete(routes[heavy],i)
        directions[heavy] = np.delete(directions[heavy],i)
        costs, _ = _route_statistics(routes,directions,workspace.travel,workspace.processing)
        position = _fragment_position(np.array([node],np.int64),np.array([bit],np.int64),
            routes,directions,np.zeros(4),costs,heavy,workspace.travel,workspace.processing,
            workspace.eligible,workspace.parent_index,workspace.points,workspace.parameters)
        if position[0] < 0:
            return original
        _put_fragment(np.array([node],np.int64),np.array([bit],np.int64),routes,directions,position)
    elif rng.random() < settings.swap_probability:
        others = [r for r in range(4) if r != heavy and len(routes[r])]
        if others:
            other = rng.choice(others)
            j = rng.randrange(len(routes[other]))
            n2 = int(routes[other][j])
            if workspace.eligible[node,other] and workspace.eligible[n2,heavy]:
                routes[heavy][i], routes[other][j] = n2,node
                directions[heavy][i], directions[other][j] = directions[other][j],bit
    if (not _heads_legal(routes,directions,workspace.points,workspace.parameters) or
        any(workspace.parent_index[a] == workspace.parent_index[b] for route in routes for a,b in zip(route,route[1:]))):
        return original
    return routes,directions


def adjacency_crossover(first, second, workspace, rng, settings):
    selected = np.array([a if rng.random() < .5 else b for a,b in zip(first.selected,second.selected)],np.int64)
    if rng.random() < settings.pattern_mutation_probability:
        parents = [i for i,options in enumerate(workspace.options) if len(options)>1]
        if parents:
            p = rng.choice(parents)
            selected[p] = rng.choice([o for o in workspace.options[p] if o != selected[p]])
    fragments, arcs = adjacency_fragments(first,second,selected,workspace,rng,settings)
    routes,directions = assemble_fragments(fragments,workspace,rng)
    routes,directions = mutate_child(routes,directions,workspace,rng,settings)
    validate_individual(workspace,routes,directions,selected)
    inherited = novel = 0
    for route,bits in zip(routes,directions):
        for k in range(1,len(route)):
            arc = (int(route[k-1]),int(bits[k-1]),int(route[k]),int(bits[k]))
            inherited += int(arc in arcs)
            novel += int(arc not in arcs)
    return routes,directions,selected,inherited,novel


def select_population(population, size):
    population = list({p.identity:p for p in population}.values())
    order = sorted(range(len(population)),key=lambda i: population[i].key)
    elite = order[0]
    quality_rank = {index:rank for rank,index in enumerate(order)}
    diversity = _population_diversity([p.edges for p in population])
    dr = sorted(range(len(population)),key=lambda i: (-diversity[i],population[i].key))
    diversity_rank = {index:rank for rank,index in enumerate(dr)}
    others = sorted((i for i in order if i != elite),key=lambda i: (
        not math.isfinite(population[i].native_cmax),quality_rank[i]+.5*diversity_rank[i],quality_rank[i]))
    return [population[i] for i in [elite]+others[:size-1]]


def final_shortlist(individuals):
    ordered = sorted(individuals,key=lambda p:p.key)
    feasible = [p for p in ordered if math.isfinite(p.native_cmax)]
    pool = (feasible or ordered)[:12]
    chosen = pool[:1]
    while len(chosen) < 3 and len(chosen) < len(pool):
        def distance(p):
            edges = set(p.edges)
            return min(1-len(edges.intersection(q.edges))/len(edges.union(q.edges)) for q in chosen)
        chosen.append(min((p for p in pool if all(p.identity != q.identity for q in chosen)),
                          key=lambda p:(-distance(p),p.key)))
    return chosen


def run_saeax(parents, config=None, settings=None, *, seed=20261110,
              scope=FORMAL_SCOPE_V2, progress_callback=None):
    config, settings = config or ScientificConfig(), settings or SAEAXConfig()
    started = time.perf_counter()
    rng = random.Random(seed)
    workspace = _HGAWorkspace(parents,config,scope)
    accounting = dict(native_calls=0,b32_calls=0,duplicate_children=0,invalid_children=0,
        construction_attempts=0,initialization_evaluations=0,native_feasible_children=0,
        crossover_s=0.,native_s=0.,population_s=0.,b32_s=0.,initialization_s=0.,
        inherited_edges=0,new_edges=0,best_improvements=0)
    seen, population, progress = {},[],[]

    def evaluate(routes,directions,selected,inherited=0,novel=0):
        validate_individual(workspace,routes,directions,selected)
        identity = complete_identity(routes,directions,selected)
        if identity in seen:
            return None
        t = time.perf_counter()
        value, _, _, _ = _priority_arrays(routes,workspace.travel,workspace.processing,
                                        workspace.points,workspace.parameters,directions)
        accounting['native_calls'] += 1
        accounting['native_s'] += time.perf_counter()-t
        if math.isnan(value):
            raise ArithmeticError('native scorer returned NaN')
        costs,loads = _route_statistics(routes,directions,workspace.travel,workspace.processing)
        edges = _numeric_edges([2*r+d for r,d in zip(routes,directions)],2*len(workspace.ids))
        item = Individual(routes,directions,selected,float(value),costs,loads,identity,edges,inherited,novel)
        seen[identity] = item
        return item

    # Only a basic geometric seed is reused. No WAG search, HGA VND, saved
    # solutions, learned model, or B32 result enters the initial population.
    reference = initialize_hga_region_seed(parents,config,scope=scope)
    selected = np.array([workspace.patterns.index(p) for p in reference.patterns],np.int64)
    routes = workspace.encode(reference)
    bits = [_greedy_route_directions(r,workspace.travel) for r in routes]
    routes,bits = _repair_oriented_first(routes,bits,workspace.points,workspace.travel,
                                       workspace.parameters,len(workspace.ids))
    initial = evaluate(routes,bits,selected)
    population.append(initial)
    attempts = 0
    while len(population) < settings.population_size and attempts < settings.population_size*settings.retries_per_child:
        attempts += 1
        try:
            r,d,s,inherit,novel = adjacency_crossover(initial,rng.choice(population),workspace,rng,settings)
            item = evaluate(r,d,s,inherit,novel)
            if item is not None:
                population.append(item)
        except ValueError:
            accounting['invalid_children'] += 1
    accounting['initialization_s'] = time.perf_counter()-started
    accounting['initialization_attempts'] = attempts
    accounting['initialization_evaluations'] = len(seen)
    best = min(population,key=lambda p:p.key)
    accounting['initial_native_cmax_s'] = best.native_cmax if math.isfinite(best.native_cmax) else None
    accounting['initial_max_processing_s'] = float(max(best.processing))
    accounting['initial_move_s'] = float(sum(best.costs-best.processing))
    generations = children = stale = 0
    termination = 'MAX_GENERATIONS'

    def snapshot():
        row = dict(generation=generations,native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
            elapsed_s=time.perf_counter()-started,valid_children=children,native_calls=accounting['native_calls'],
            diversity=float(np.mean(_population_diversity([p.edges for p in population]))),
            max_processing_s=float(max(best.processing)),move_s=float(sum(best.costs-best.processing)),
            feasible_population=sum(math.isfinite(p.native_cmax) for p in population),
            duplicate_children=accounting['duplicate_children'])
        progress.append(row)
        if progress_callback is not None:
            progress_callback(row)

    snapshot()
    while generations < settings.max_generations:
        batch = []
        old_best = best.native_cmax
        for _ in range(settings.offspring_per_generation*settings.retries_per_child):
            if len(batch) == settings.offspring_per_generation:
                break
            accounting['construction_attempts'] += 1
            first = min(rng.sample(population,min(2,len(population))),key=lambda p:p.key)
            other = [p for p in population if p.identity != first.identity]
            second = min(rng.sample(other,min(2,len(other))),key=lambda p:p.key) if other else first
            t = time.perf_counter()
            try:
                r,d,s,inherit,novel = adjacency_crossover(first,second,workspace,rng,settings)
            except ValueError:
                accounting['crossover_s'] += time.perf_counter()-t
                accounting['invalid_children'] += 1
                continue
            accounting['crossover_s'] += time.perf_counter()-t
            item = evaluate(r,d,s,inherit,novel)
            if item is None:
                accounting['duplicate_children'] += 1
                continue
            batch.append(item)
            children += 1
            accounting['native_feasible_children'] += int(math.isfinite(item.native_cmax))
            accounting['inherited_edges'] += inherit
            accounting['new_edges'] += novel
            if item.key < best.key:
                if item.native_cmax < best.native_cmax-1.e-8:
                    accounting['best_improvements'] += 1
                best = item
        t = time.perf_counter()
        population = select_population(population+batch+[best],settings.population_size)
        accounting['population_s'] += time.perf_counter()-t
        if len(batch) != settings.offspring_per_generation:
            termination = 'NOVEL_CHILD_RETRIES_EXHAUSTED'
            break
        generations += 1
        stale = 0 if best.native_cmax < old_best-1.e-8 else stale+1
        if generations % 25 == 0 or generations in (100,200,400):
            snapshot()
        if generations >= settings.min_generations and stale >= settings.stagnation_generations:
            termination = 'STAGNATION'
            break
    if progress[-1]['generation'] != generations:
        snapshot()
    accounting['stagnation_at_stop'] = stale
    accounting['population_size_at_stop'] = len(population)
    accounting['search_s'] = time.perf_counter()-started
    final, selected_final, final_rows, fallback = None,None,[],None
    for item in final_shortlist(seen.values()):
        solution = workspace.solution(item.routes,item.selected)
        expected = tuple(tuple(workspace.ids[n] for n in route) for route in item.routes)
        if (tuple(r.block_ids for r in solution.routes) != expected or
            solution.patterns != tuple(workspace.patterns[p] for p in item.selected)):
            raise RuntimeError('canonicalization changed the directed individual')
        vectors = tuple(tuple(int(d) for d in bits) for bits in item.directions)
        t = time.perf_counter()
        accounting['b32_calls'] += 1
        schedule = reference_schedule_formal(solution,config,scope=scope,orientations=dict(enumerate(vectors)))
        if fallback is None:
            fallback = (solution,vectors,item,schedule)
        elapsed = time.perf_counter()-t
        accounting['b32_s'] += elapsed
        final_rows.append(dict(rank=len(final_rows)+1,native_cmax_s=item.native_cmax if math.isfinite(item.native_cmax) else None,
            status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,b32_s=elapsed,
            max_processing_s=float(max(item.processing)),move_s=float(sum(item.costs-item.processing)),
            wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
            diagnostic=' | '.join(schedule.diagnostics[:2])))
        if schedule.feasible and (final is None or schedule.cmax < final.cmax):
            if schedule.directions != vectors:
                raise RuntimeError('B32 changed explicit directions')
            final, selected_final = schedule,(solution,vectors,item)
    accounting['native_best_cmax_s'] = best.native_cmax if math.isfinite(best.native_cmax) else None
    if selected_final is None:
        solution,vectors,item,failed_schedule = fallback
        accounting['processing_s'] = [float(p) for p in item.processing]
        accounting['move_s'] = float(sum(item.costs-item.processing))
        accounting['wait_s'] = sum(op.duration for op in failed_schedule.operations if op.kind is OperationKind.WAIT)
        accounting['wait_is_partial'] = True
        return SAEAXResult('DEADLOCK',None,solution,vectors,failed_schedule,accounting['native_best_cmax_s'],
            generations,children,termination,time.perf_counter()-started,accounting,progress,final_rows)
    solution,vectors,item = selected_final
    accounting['processing_s'] = [float(p) for p in item.processing]
    accounting['move_s'] = float(sum(item.costs-item.processing))
    accounting['wait_s'] = sum(op.duration for op in final.operations if op.kind is OperationKind.WAIT)
    return SAEAXResult('FEASIBLE',final.cmax,solution,vectors,final,
        item.native_cmax if math.isfinite(item.native_cmax) else None,generations,children,
        termination,time.perf_counter()-started,accounting,progress,final_rows)
