"""Joint parent-set selection and directed arc MILP, with NATIVE-only acceptance.

This model contains task/direction vertices, not routes/columns. Each robot has
one source-to-sink path. Unselected tasks are mandatory fixed-direction vertices
whose original relative order is imposed. Selected tasks have optional vertices
for every legal mode, eligible robot and direction. Mode-linked degree rows,
MTZ positions and fixed-task order rows enforce exact coverage and connectivity.
Only the route-period proxy is optimized; the original NATIVE decides acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch.nn import functional as F

from mrta_baselines.hga import _HGAWorkspace
from mrta_reference.model import ScientificConfig, OperationKind
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map
from mrta_reference.scheduler import native_schedule, selected_schedule_wait_blockers
from mrta_search import saeax
from mrta_search.saeax import _heads_legal, _route_statistics, complete_identity, validate_individual
from mrta_search.gnn_constructor import VNextGNN, joint_graph, build_block_robot_graph
from mrta_search.lns import oriented_route_state, schedule_feedback_summary, actual_parent_transfers
from mrta_search.coordination_risk import coordination_risk


@dataclass(frozen=True)
class JNConfig:
    max_rounds: int = 2
    sets_per_round: int = 2
    pool_size: int = 12
    mip_seconds: float = 5.
    final_coordination_budget: int = 4
    training_sets: int = 4
    training_epochs: int = 6


@dataclass
class Plan:
    solution: object
    directions: tuple
    routes: list
    bits: list
    selected: np.ndarray
    costs: np.ndarray
    loads: np.ndarray
    schedule: object = None
    blockers: tuple = ()

    @property
    def key(self):
        return complete_identity(self.routes, self.bits, self.selected)

    @property
    def period(self):
        return float(max(self.costs))


def make_plan(w, routes, bits, selected, *, schedule=None, blockers=()):
    rr = [np.array(row, np.int64) for row in routes]
    dd = [np.array(row, np.int64) for row in bits]
    ss = np.array(selected, np.int64)
    validate_individual(w, rr, dd, ss)
    if not _heads_legal(rr, dd, w.points, w.parameters):
        raise ValueError('joint MILP returned illegal initial heads')
    solution = w.solution(rr, ss)
    expected = tuple(tuple(w.ids[int(n)] for n in row) for row in rr)
    if (tuple(r.block_ids for r in solution.routes) != expected or
            solution.patterns != tuple(w.patterns[m] for m in ss)):
        raise ValueError('canonicalization changed the directed joint solution')
    vectors = tuple(tuple(map(int, row)) for row in dd)
    costs, loads = _route_statistics(rr, dd, w.travel, w.processing)
    return Plan(solution, vectors, rr, dd, ss, costs, loads, schedule, tuple(blockers))


def plan_from_initial(w, initial, blockers=()):
    return make_plan(w, w.encode(initial.solution), initial.directions,
        [w.patterns.index(p) for p in initial.solution.patterns],
        schedule=initial.schedule, blockers=blockers)


def initialize(parents, seed=20261110, progress_callback=None):
    """Observe the SAME SAEAX NATIVE calls, never replay its selected initial plan.

    The existing SAEAX interface has no NATIVE-observer parameter. A temporary
    single-process observer adapter forwards each original call unchanged and
    is restored immediately. No scheduler/core file is modified.
    """
    started = time.perf_counter()
    original = saeax.native_schedule
    orders = {}
    def observe(*args, **kwargs):
        def capture(candidate, order):
            orders[candidate.operations] = tuple(order)
        return original(*args, **kwargs, feasible_dispatch_observer=capture)
    with patch.object(saeax, 'native_schedule', observe):
        result = saeax.run_saeax(parents, seed=seed, progress_callback=progress_callback)
    blockers = ()
    if result.status == 'FEASIBLE':
        blockers = selected_schedule_wait_blockers(result.solution, result.directions,
            result.schedule, orders[result.schedule.operations], ScientificConfig())
    return result, blockers, time.perf_counter()-started


class _Rows:
    def __init__(self):
        self.i, self.j, self.v, self.lower, self.upper = [], [], [], [], []

    def add(self, terms, lower=-np.inf, upper=np.inf):
        row = len(self.lower)
        for col, value in terms:
            if value:
                self.i.append(row); self.j.append(col); self.v.append(float(value))
        self.lower.append(lower); self.upper.append(upper)

    def matrix(self, variables):
        return coo_matrix((self.v, (self.i, self.j)), shape=(len(self.lower), variables)).tocsc()


class JointArcModel:
    """All insertion gaps, all legal Q modes, explicit directions, four paths.

    A fixed->fixed arc exists only for consecutive retained tasks. Optional
    vertices can connect to any retained task or any OTHER selected parent.
    Together with mandatory fixed-task visits and increasing fixed positions,
    this expresses every interleaving preserving the retained relative order.
    """
    SOURCE, SINK = -1, -2

    def __init__(self, w, current, parents):
        self.w, self.current = w, current
        self.parents = tuple(sorted(set(map(int, parents))))
        if not self.parents or any(p < 0 or p >= len(w.parents) for p in self.parents):
            raise ValueError('invalid joint parent set')
        selected_set = set(self.parents)
        self.lower, self.upper, self.integer = [], [], []
        def variable(low=0., high=1., integer=1):
            index = len(self.lower)
            self.lower.append(low); self.upper.append(high); self.integer.append(integer)
            return index
        self.mode = {int(m): variable() for p in self.parents for m in w.options[p]}
        self.vertices, self.arcs, self.position, self.visit = [], [], [], []
        self.fixed = []
        for r in range(4):
            retained = [(int(n), int(d)) for n, d in zip(current.routes[r], current.bits[r])
                        if int(w.parent_index[n]) not in selected_set]
            optional = [(int(n), d) for p in self.parents for m in w.options[p]
                for n in w.pattern_nodes[m] if n >= 0 and w.eligible[n, r] for d in (0, 1)]
            vertices = retained+optional
            self.vertices.append(vertices); self.fixed.append(len(retained))
            maximum = len(retained)+2*len(self.parents)
            self.visit.append([variable(1. if i < len(retained) else 0.) for i in range(len(vertices))])
            self.position.append([variable(i+1 if i < len(retained) else 0.,
                i+1+2*len(self.parents) if i < len(retained) else maximum, 0)
                for i in range(len(vertices))])
            arcs = {}
            def arc(a, b):
                if a >= 0 and b >= 0 and w.parent_index[vertices[a][0]] == w.parent_index[vertices[b][0]]:
                    return
                arcs[a, b] = variable()
            if not retained:
                arc(self.SOURCE, self.SINK)
            else:
                arc(self.SOURCE, 0); arc(len(retained)-1, self.SINK)
                for i in range(len(retained)-1):
                    arc(i, i+1)
            for j in range(len(retained), len(vertices)):
                arc(self.SOURCE, j); arc(j, self.SINK)
                for i in range(len(retained)):
                    arc(i, j); arc(j, i)
                for i in range(len(retained), len(vertices)):
                    if i != j:
                        arc(i, j)
            self.arcs.append(arcs)
        self.z = variable(0., current.period+1.e-6, 0)
        rows = _Rows()
        for p in self.parents:
            rows.add([(self.mode[int(m)], 1.) for m in w.options[p]], 1., 1.)
            for m in w.options[p]:
                for n in w.pattern_nodes[m]:
                    if n < 0:
                        continue
                    terms = [(self.mode[int(m)], -1.)]
                    for r in range(4):
                        terms.extend((self.visit[r][i], 1.) for i, (node, d) in enumerate(self.vertices[r]) if node == n)
                    rows.add(terms, 0., 0.)
        for r, vertices in enumerate(self.vertices):
            arcs, visit, position = self.arcs[r], self.visit[r], self.position[r]
            maximum = self.fixed[r]+2*len(self.parents)
            incoming = [[] for _ in vertices]; outgoing = [[] for _ in vertices]
            for (i, j), x in arcs.items():
                if i >= 0: outgoing[i].append((x, 1.))
                if j >= 0: incoming[j].append((x, 1.))
                if i >= 0 and j >= 0:
                    rows.add([(position[i], 1.), (position[j], -1.), (x, maximum+1.)], upper=maximum)
            rows.add([(x, 1.) for (i, j), x in arcs.items() if i == self.SOURCE], 1., 1.)
            rows.add([(x, 1.) for (i, j), x in arcs.items() if j == self.SINK], 1., 1.)
            for i in range(len(vertices)):
                rows.add(incoming[i]+[(visit[i], -1.)], 0., 0.)
                rows.add(outgoing[i]+[(visit[i], -1.)], 0., 0.)
                rows.add([(position[i], 1.), (visit[i], -maximum)], upper=0.)
                rows.add([(position[i], 1.), (visit[i], -1.)], lower=0.)
            for i in range(self.fixed[r]-1):
                rows.add([(position[i], 1.), (position[i+1], -1.)], upper=-1.)
            load = [(self.z, -1.)]+[(visit[i], w.processing[n]) for i, (n, d) in enumerate(vertices)]
            for (i, j), x in arcs.items():
                if i >= 0 and j >= 0:
                    a, d = vertices[i]; b, e = vertices[j]
                    load.append((x, w.travel[a, b, d, e]))
            rows.add(load, upper=0.)
        self.head_conflicts = 0
        for r in range(4):
            for s in range(r+1, 4):
                for (a, i), x in self.arcs[r].items():
                    if a != self.SOURCE or i < 0: continue
                    for (b, j), y in self.arcs[s].items():
                        if b != self.SOURCE or j < 0: continue
                        rr = [np.empty(0, np.int64) for _ in range(4)]
                        dd = [np.empty(0, np.int64) for _ in range(4)]
                        n, d = self.vertices[r][i]; m, e = self.vertices[s][j]
                        rr[r], dd[r] = np.array([n], np.int64), np.array([d], np.int64)
                        rr[s], dd[s] = np.array([m], np.int64), np.array([e], np.int64)
                        if not _heads_legal(rr, dd, w.points, w.parameters):
                            rows.add([(x, 1.), (y, 1.)], upper=1.); self.head_conflicts += 1
        self.matrix = rows.matrix(len(self.lower))
        self.constraints = LinearConstraint(self.matrix, np.array(rows.lower), np.array(rows.upper))
        self.objective = np.zeros(len(self.lower)); self.objective[self.z] = 1.

    def encode(self, plan):
        """A concrete feasible incumbent witness for tests/debugging, not a solver."""
        result = np.zeros(len(self.lower))
        for m, var in self.mode.items():
            result[var] = float(m in plan.selected)
        for r, vertices in enumerate(self.vertices):
            mapping = {pair: i for i, pair in enumerate(vertices)}
            previous = self.SOURCE
            for order, pair in enumerate(zip(map(int, plan.routes[r]), map(int, plan.bits[r])), 1):
                i = mapping[pair]
                result[self.visit[r][i]] = 1.; result[self.position[r][i]] = order
                result[self.arcs[r][previous, i]] = 1.; previous = i
            result[self.arcs[r][previous, self.SINK]] = 1.
        result[self.z] = plan.period
        return result

    def decode(self, values):
        binary = np.flatnonzero(self.integer)
        if max(np.abs(values[binary]-np.rint(values[binary])), default=0.) > 1.e-5:
            raise ValueError('MILP incumbent has fractional discrete variables')
        selected = self.current.selected.copy()
        for p in self.parents:
            options = [int(m) for m in self.w.options[p] if values[self.mode[int(m)]] > .5]
            if len(options) != 1: raise ValueError('MILP parent mode count')
            selected[p] = options[0]
        routes, bits = [], []
        for r, vertices in enumerate(self.vertices):
            successor = {i: j for (i, j), x in self.arcs[r].items() if values[x] > .5}
            seen, route, vector, i = set(), [], [], self.SOURCE
            while True:
                if i not in successor: raise ValueError('MILP disconnected path')
                j = successor[i]
                if j == self.SINK: break
                if j in seen or len(seen) >= len(vertices): raise ValueError('MILP subcycle')
                seen.add(j); n, d = vertices[j]; route.append(n); vector.append(d); i = j
            active = {j for j, x in enumerate(self.visit[r]) if values[x] > .5}
            if seen != active: raise ValueError('MILP active tasks outside source path')
            retained = [(n, d) for n, d in zip(route, vector) if int(self.w.parent_index[n]) not in self.parents]
            if retained != vertices[:self.fixed[r]]: raise ValueError('unselected route order/direction changed')
            routes.append(route); bits.append(vector)
        plan = make_plan(self.w, routes, bits, selected)
        if plan.period > values[self.z]+1.e-4: raise ValueError('MILP cost differs from formal route cost')
        return plan


def solve_joint(w, current, parents, seconds=5.):
    started = time.perf_counter()
    model = JointArcModel(w, current, parents)
    built = time.perf_counter()
    result = milp(model.objective, integrality=np.array(model.integer),
        bounds=Bounds(np.array(model.lower), np.array(model.upper)), constraints=model.constraints,
        options=dict(time_limit=seconds, mip_rel_gap=0.))
    elapsed = time.perf_counter()-built
    def finite(value):
        return float(value) if value is not None and math.isfinite(value) else None
    record = dict(parents=[w.parents[p].parent_id for p in model.parents],
        status=int(result.status), message=str(result.message), has_incumbent=result.x is not None,
        objective_s=finite(result.fun), dual_bound_s=finite(getattr(result, 'mip_dual_bound', None)),
        mip_gap=finite(getattr(result, 'mip_gap', None)), build_s=built-started, solve_s=elapsed,
        variables=len(model.lower), constraints=model.matrix.shape[0],
        arcs=sum(map(len, model.arcs)), head_conflicts=model.head_conflicts)
    plan = model.decode(result.x) if result.x is not None else None
    if plan is not None: record['route_period_s'] = plan.period
    return plan, record


def evaluate_plan(plan, w):
    orders = {}
    started = time.perf_counter()
    schedule = native_schedule(plan.solution, w.config, scope=FORMAL_SCOPE_V2,
        orientations=dict(enumerate(plan.directions)),
        feasible_dispatch_observer=lambda candidate, order: orders.setdefault(candidate.operations, tuple(order)))
    elapsed = time.perf_counter()-started
    plan.schedule = schedule
    if schedule.feasible:
        if schedule.directions != plan.directions: raise ValueError('NATIVE changed joint directions')
        plan.blockers = selected_schedule_wait_blockers(plan.solution, plan.directions,
            schedule, orders[schedule.operations], w.config)
    return elapsed


PARENT_DIM, SET_DIM = 30, 72


class SetSelector(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder or VNextGNN().enable_conflict_graph()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        hidden = self.encoder.layer2.bias.numel()
        self.head = nn.Sequential(nn.Linear(4*hidden+SET_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, rows):
        return self.head(rows).squeeze(-1)


def load_selector(checkpoint, seed=20261110):
    torch.manual_seed(seed)
    encoder = VNextGNN().enable_conflict_graph()
    saved = torch.load(checkpoint, weights_only=False, map_location='cpu')['state_dict']
    keys = {k for k in encoder.state_dict() if k.startswith(('layer1.', 'layer2.', 'geometry_skip.'))}
    state = {k: v for k, v in saved.items() if k in keys}
    if set(state) != keys: raise ValueError('incomplete existing geometry encoder')
    encoder.load_state_dict(state, strict=False); encoder.eval()
    return SetSelector(encoder)


class SelectionContext:
    """Only BEFORE-neighborhood state and observed selected NATIVE timeline."""
    def __init__(self, w, current, selector=None):
        self.w, self.current, self.selector = w, current, selector
        self.critical = int(np.argmax(current.costs))
        self.scale = max(1., current.period)
        self.feedback = schedule_feedback_summary(current.solution, current.schedule, current.blockers, w.config)
        self.owners = [set() for _ in w.parents]
        processing = np.zeros((len(w.parents), 4)); moves = processing.copy(); positions = processing.copy()
        counts = processing.copy()
        for r, (route, bits) in enumerate(zip(current.routes, current.bits)):
            for k, n in enumerate(route):
                p = int(w.parent_index[n]); self.owners[p].add(r)
                processing[p, r] += w.processing[n]
                positions[p, r] += k/max(1, len(route)-1); counts[p, r] += 1
                if k:
                    cost = w.travel[route[k-1], n, bits[k-1], bits[k]]
                    moves[p, r] += .5*cost; moves[w.parent_index[route[k-1]], r] += .5*cost
        positions /= np.maximum(1., counts)
        self.processing, self.moves = processing, moves
        self.possible = np.array([np.any(w.eligible[[int(n) for m in options for n in w.pattern_nodes[m] if n >= 0]], axis=0)
            for options in w.options])
        numeric = []
        for p, parent in enumerate(w.parents):
            work = sum(processing[p])
            receivers = [(current.costs[r]+work)/self.scale if self.possible[p, r] else 2. for r in range(4)]
            kinds = [sum(w.patterns[m].kind.value == kind for m in w.options[p])/len(w.options[p])
                     for kind in ('WHOLE', 'Y_SPLIT', 'X_SPLIT')]
            numeric.append([*(processing[p]/self.scale), *(moves[p]/self.scale),
                *self.possible[p].astype(float), *positions[p], *receivers,
                self.feedback['parent_wait'].get(parent.parent_id, 0.)/self.scale,
                self.feedback['parent_blocked'].get(parent.parent_id, 0.)/self.scale,
                processing[p, self.critical]/self.scale, work/self.scale,
                (parent.start[0]+parent.end[0])/40., (parent.start[1]+parent.end[1])/24.,
                *kinds, len(self.owners[p])/4.])
        self.numeric = np.array(numeric).reshape(-1, PARENT_DIM)
        self.rule_parent = (processing[:, self.critical]+.25*moves.sum(1))/self.scale+self.numeric[:, 20]+self.numeric[:, 21]
        self.risk = coordination_risk(current.solution, w.config).total/20.
        self.parent_embeddings = self.robot_embeddings = None
        self.blocking_edges = 0
        if selector is not None:
            graph, blocks, _ = joint_graph(current.solution, w.config)
            nodes = dict(graph.block_index)
            pairs = [(event.delayed_block_id, block) for event in current.blockers
                     for _, block, intervals in event.blocker_intervals if intervals]
            feedback = dict(self.feedback, blocked_pairs=pairs)
            state = oriented_route_state(tuple(r.block_ids for r in current.solution.routes),
                current.directions, blocks, w.config)
            dynamic = build_block_robot_graph((), (), w.config, base_graph=graph,
                state=state, block_nodes=nodes, feedback=feedback)
            self.blocking_edges = sum(int(t) in (8, 9) for t in dynamic.edge_type)
            with torch.no_grad(): z = selector.encoder.encode_static(dynamic).detach()
            self.parent_embeddings = torch.stack([torch.stack([z[nodes[b.block_id]]
                for b in graph.blocks if b.parent_id == p.parent_id]).mean(0) for p in w.parents])
            self.robot_embeddings = z[graph.robot_offset:graph.robot_offset+4]

    def candidate_sets(self, limit=12):
        critical = [p for p in range(len(self.w.parents)) if self.critical in self.owners[p]]
        critical.sort(key=lambda p: (-self.rule_parent[p], p))
        all_ranked = sorted(range(len(self.w.parents)), key=lambda p: (-self.rule_parent[p], p))
        result = []
        for k in (4, 5, 6):
            for offset in range(max(4, len(critical))):
                if not critical: break
                anchor = critical[offset % len(critical)]
                others = [p for p in all_ranked if self.owners[p]-{self.critical}]
                others.sort(key=lambda p: (-int(any(self.possible[anchor, r] for r in self.owners[p]-{self.critical})),
                    -sum(self.processing[p, r] for r in self.owners[p]-{self.critical}), p))
                chosen = [anchor]
                if others: chosen.append(others[offset % len(others)])
                tail = critical[(offset+1) % len(critical):]+critical[:(offset+1) % len(critical)]
                tail += all_ranked[offset % max(1, len(all_ranked)):]+all_ranked[:offset % max(1, len(all_ranked))]
                for p in tail:
                    if p not in chosen: chosen.append(p)
                    if len(chosen) >= k: break
                key = tuple(sorted(chosen))
                robots = set().union(*(self.owners[p] for p in key))
                if len(key) == k and len(robots) >= 2 and key not in result: result.append(key)
                if len(result) >= (k-3)*max(1, limit//3): break
        return result[:limit]

    def rule_score(self, parents):
        robots = set().union(*(self.owners[p] for p in parents))
        return float(sum(self.rule_parent[list(parents)])+.01*len(robots))

    def rows(self, sets):
        result = []
        for parents in sets:
            indices = list(parents)
            mask = np.array([any(r in self.owners[p] for p in parents) for r in range(4)], float)
            parent_rows = self.numeric[indices]
            receiver_slack = sum(max(0., self.scale-self.current.costs[r]) for r in range(4)
                if any(self.possible[p, r] for p in parents))/self.scale
            numeric = np.r_[parent_rows.mean(0), parent_rows.max(0), self.current.costs/self.scale,
                mask, len(parents)/6., sum(self.critical in self.owners[p] for p in parents)/len(parents),
                self.risk, receiver_slack]
            h = self.parent_embeddings[indices]
            receiver = np.array([any(self.possible[p, r] for p in parents) for r in range(4)])
            result.append(torch.cat((h.mean(0), h.max(0).values, self.robot_embeddings[self.critical],
                self.robot_embeddings[receiver].mean(0), torch.tensor(numeric, dtype=torch.float32))))
        return torch.stack(result) if result else torch.empty((0, 4*64+SET_DIM))


def changes(w, before, after):
    return dict(parent_modes=[w.parents[p].parent_id for p in range(len(w.parents))
        if before.selected[p] != after.selected[p]],
        transfers=actual_parent_transfers(before, after, w.config))


def solution_record(plan):
    if plan is None: return None
    return dict(patterns=[dict(parent_id=p.parent_id, kind=p.kind.value, t=p.t,
        point_id=p.point_id, rail=p.rail.value if p.rail else None) for p in plan.solution.patterns],
        routes=[list(r.block_ids) for r in plan.solution.routes], directions=plan.directions)


def trial(w, current, parents, cache, settings):
    candidate, record = solve_joint(w, current, parents, settings.mip_seconds)
    record.update(coordination_called=False, coordination_s=0., coordination_status=None, coordination_cmax_s=None,
        old_cmax_s=current.schedule.cmax, actual_gain_s=None)
    if candidate is None: return None, record
    record['changes'] = changes(w, current, candidate)
    if candidate.key in cache:
        cached = cache[candidate.key]
        candidate.schedule, candidate.blockers = cached.schedule, cached.blockers
    else:
        record['coordination_s'] = evaluate_plan(candidate, w); record['coordination_called'] = True
        cache[candidate.key] = candidate
    record.update(coordination_status=candidate.schedule.status.value,
        coordination_cmax_s=candidate.schedule.cmax if candidate.schedule.feasible else None,
        actual_gain_s=current.schedule.cmax-candidate.schedule.cmax if candidate.schedule.feasible else None,
        solution=solution_record(candidate))
    return candidate, record


def collect_labels(w, current, selector, settings=None):
    settings = settings or JNConfig()
    context = SelectionContext(w, current, selector)
    pool = context.candidate_sets(settings.pool_size)
    chosen = pool[::max(1, len(pool)//settings.training_sets)][:settings.training_sets]
    x = context.rows(chosen).detach()
    cache, records = {current.key: current}, []
    for parents in chosen:
        _, record = trial(w, current, parents, cache, settings); records.append(record)
    return x, records


def fit_selector(selector, groups, epochs=6):
    """Only actual MILP+NATIVE gains determine preferences; no proxy labels."""
    started = time.perf_counter()
    positives = sum(r.get('actual_gain_s') is not None and r['actual_gain_s'] > 1.e-6 for _, records in groups for r in records)
    pairs = []
    for x, records in groups:
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                a, b = records[i], records[j]
                if a.get('coordination_status') != 'FEASIBLE' or b.get('coordination_status') != 'FEASIBLE': continue
                delta = a['actual_gain_s']-b['actual_gain_s']
                if abs(delta) > 1.e-6: pairs.append((x[i], x[j], 1. if delta > 0 else -1.))
    if not positives or not pairs:
        return dict(trained=False, reason='NO_POSITIVE_NATIVE_RECONSTRUCTION' if not positives else 'NO_NONFLAT_NATIVE_PREFERENCES',
            positive_labels=positives, preference_pairs=len(pairs), epochs=0, runtime_s=time.perf_counter()-started)
    a = torch.stack([p[0] for p in pairs]); b = torch.stack([p[1] for p in pairs]); sign = torch.tensor([p[2] for p in pairs])
    optimizer = torch.optim.Adam(selector.head.parameters(), lr=.003)
    losses, gradient = [], 0.
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.softplus(-sign*(selector(a)-selector(b))).mean(); loss.backward()
        gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in selector.head.parameters() if p.grad is not None)))
        optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad(): accuracy = float((sign*(selector(a)-selector(b)) > 0).float().mean())
    selector.eval()
    return dict(trained=True, positive_labels=positives, preference_pairs=len(pairs), epochs=epochs,
        loss=losses, head_gradient_norm=gradient, training_rank_accuracy=accuracy,
        runtime_s=time.perf_counter()-started, target='actual_NATIVE_Cmax_improvement_after_joint_MILP')


@dataclass
class JNResult:
    plan: Plan
    runtime: float
    reason: str
    accounting: dict
    records: list = field(default_factory=list)


def optimize(w, initial_plan, initialization_s, *, selector=None, settings=None, progress_callback=None):
    settings = settings or JNConfig()
    started = time.perf_counter()
    current, records, cache, solved = initial_plan, [], {initial_plan.key: initial_plan}, set()
    a = dict(rounds=0, milp_calls=0, coordination_calls=0, accepted=0, gnn_scored_sets=0,
        gnn_choice_differences=0, initialization_s=initialization_s,
        initial_cmax_s=initial_plan.schedule.cmax, initial_route_period_s=initial_plan.period)
    reason = 'ROUND_LIMIT'
    for iteration in range(settings.max_rounds):
        a['rounds'] += 1
        context = SelectionContext(w, current, selector)
        pool = [q for q in context.candidate_sets(settings.pool_size) if (current.key, q) not in solved]
        rule_order = sorted(range(len(pool)), key=lambda i: (-context.rule_score(pool[i]), pool[i]))
        if selector is None: order = rule_order
        else:
            with torch.no_grad(): scores = selector(context.rows(pool)).tolist()
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]))
            a['gnn_scored_sets'] += len(pool)
            a['gnn_choice_differences'] += int(order[:settings.sets_per_round] != rule_order[:settings.sets_per_round])
        if not pool: reason = 'NO_NEW_PARENT_SETS'; break
        best = current
        for i in order[:settings.sets_per_round]:
            if a['coordination_calls'] >= settings.final_coordination_budget: reason = 'NATIVE_BUDGET'; break
            parents = pool[i]; solved.add((current.key, parents))
            candidate, record = trial(w, current, parents, cache, settings)
            a['milp_calls'] += 1; a['coordination_calls'] += int(record['coordination_called'])
            record['iteration'] = iteration+1; records.append(record)
            if candidate is not None and candidate.schedule.feasible and candidate.schedule.cmax < best.schedule.cmax-1.e-6:
                best = candidate
            if progress_callback:
                progress_callback({k: v for k, v in record.items() if k != 'solution'})
        if best.key == current.key: reason = 'NO_NATIVE_IMPROVEMENT'; break
        current = best; a['accepted'] += 1
        if a['coordination_calls'] >= settings.final_coordination_budget: reason = 'NATIVE_BUDGET'; break
    a.update(final_cmax_s=current.schedule.cmax, final_route_period_s=current.period,
        returned_initial=current.key == initial_plan.key, changes=changes(w, initial_plan, current),
        search_complete=True, new_mip_build_s=sum(r['build_s'] for r in records),
        new_mip_s=sum(r['solve_s'] for r in records), new_coordination_s=sum(r['coordination_s'] for r in records))
    return JNResult(current, initialization_s+time.perf_counter()-started, reason, a, records)
