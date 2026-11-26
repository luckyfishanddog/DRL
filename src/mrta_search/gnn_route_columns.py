"""Directed route pricing and a coupled min-max restricted master.

LP rows are sum_m y_pm=1, sum_c a_bc*x_c-y_m(b)=0,
sum_c:r(c)=r x_c=1, and sum_c:r(c)=r C_c*x_c-z<=0.
The last row strengthens C_c*x_c<=z without changing the integer problem.
Incompatible first tasks add x_i+x_j<=1, using the original head predicate.

With SciPy equality marginals (tau, pi, sigma), load marginals mu<=0,
alpha=-mu, a NEW column has rc=alpha_r*C_c-sum_b pi_b-sigma_r.
Its new incompatibility rows have zero-extended duals. Existing columns also
include their existing incompatibility-row marginals. Thus no old row's dual
is silently discarded. This is heuristic pricing with growing compatibility
rows, not an exact branch-and-price algorithm or an optimality certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack
import torch
from torch import nn
from torch.nn import functional as F

from mrta_baselines.hga import (_HGAWorkspace, _greedy_route_directions,
    _oriented_route_cost, _repair_oriented_first)
from mrta_reference.model import OperationKind, ScientificConfig
from mrta_reference.scheduler import native_schedule
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_search.gnn_constructor import VNextGNN, complete_parent_space
from mrta_search.initialization import _construct_rail_monotone_balanced_bootstrap
from mrta_search.saeax import (_heads_legal, _route_statistics, complete_identity,
    run_saeax, validate_individual)


@dataclass(frozen=True)
class RCGConfig:
    max_rounds: int = 8
    columns_per_robot_round: int = 4
    pool_per_robot: int = 80
    empty_rounds: int = 2
    final_coordination_budget: int = 2
    mip_seconds: float = 5.


@dataclass(frozen=True)
class RouteColumn:
    robot: int
    nodes: tuple[int, ...]
    directions: tuple[int, ...]
    cost: float

    @property
    def key(self):
        return self.robot, tuple(zip(self.nodes, self.directions))


@dataclass
class RouteBundle:
    columns: tuple[RouteColumn, ...]
    selected: np.ndarray

    @property
    def period(self):
        return max(c.cost for c in self.columns)

    @property
    def key(self):
        return complete_identity([c.nodes for c in self.columns],
            [c.directions for c in self.columns], self.selected)


def make_column(w, robot, nodes, directions):
    nodes, directions = tuple(map(int, nodes)), tuple(map(int, directions))
    if len(nodes) != len(directions) or any(d not in (0, 1) for d in directions):
        raise ValueError('unaligned column directions')
    if len(set(nodes)) != len(nodes) or any(not w.eligible[n, robot] for n in nodes):
        raise ValueError('column duplicate or qualification/fixed X ownership')
    chosen = {}
    for n in nodes:
        parent, mode = int(w.parent_index[n]), int(w.node_pattern[n])
        if parent in chosen and chosen[parent] != mode:
            raise ValueError('a route contains mutually exclusive parent modes')
        chosen[parent] = mode
    if any(w.parent_index[a] == w.parent_index[b] for a, b in zip(nodes, nodes[1:])):
        raise ValueError('adjacent siblings violate canonical rules')
    cost = float(_oriented_route_cost(np.array(nodes, np.int64),
        np.array(directions, np.int64), w.travel, w.processing))
    return RouteColumn(robot, nodes, directions, cost)


def make_bundle(w, routes, bits, selected):
    rr = [np.array(row, np.int64) for row in routes]
    dd = [np.array(row, np.int64) for row in bits]
    ss = np.array(selected, np.int64)
    validate_individual(w, rr, dd, ss)
    if not _heads_legal(rr, dd, w.points, w.parameters):
        raise ValueError('incompatible route heads')
    return RouteBundle(tuple(make_column(w, r, rr[r], dd[r]) for r in range(4)), ss)


def bundle_from_solution(w, solution, directions):
    selected = np.array([w.patterns.index(p) for p in solution.patterns], np.int64)
    return make_bundle(w, w.encode(solution), directions, selected)


def export_bundle(w, bundle):
    routes = [np.array(c.nodes, np.int64) for c in bundle.columns]
    bits = tuple(c.directions for c in bundle.columns)
    make_bundle(w, routes, bits, bundle.selected)
    solution = w.solution(routes, bundle.selected)
    expected = tuple(tuple(w.ids[n] for n in c.nodes) for c in bundle.columns)
    if (tuple(r.block_ids for r in solution.routes) != expected or
            solution.patterns != tuple(w.patterns[p] for p in bundle.selected)):
        raise ValueError('canonicalization changed a directed route bundle')
    return solution, bits


def heads_compatible(w, first, second):
    if not first.nodes or not second.nodes:
        return True
    routes = [np.empty(0, np.int64) for _ in range(4)]
    bits = [np.empty(0, np.int64) for _ in range(4)]
    for c in (first, second):
        routes[c.robot] = np.array(c.nodes[:1], np.int64)
        bits[c.robot] = np.array(c.directions[:1], np.int64)
    return bool(_heads_legal(routes, bits, w.points, w.parameters))


@dataclass
class DualPrices:
    parent: np.ndarray
    coverage: np.ndarray
    robot: np.ndarray
    load: np.ndarray
    existing: dict = field(default_factory=dict)

    def reduced_cost(self, column):
        if column.key in self.existing:
            return self.existing[column.key]
        # NEW columns enter old pair rows with coefficient zero; pair rows
        # created for them are assigned zero dual when extending this LP dual.
        return float(self.load[column.robot]*column.cost -
            sum(self.coverage[n] for n in column.nodes) - self.robot[column.robot])


class RestrictedMaster:
    def __init__(self, workspace, pools):
        self.w = w = workspace
        self.columns = tuple(c for pool in pools for c in pool)
        self.nx, self.ny = len(self.columns), len(w.patterns)
        self.z = self.nx+self.ny
        variables = self.z+1
        count_p, count_b = len(w.parents), len(w.ids)
        er, ec, ev, ur, uc, uv = [], [], [], [], [], []
        def eq(row, col, value):
            er.append(row); ec.append(col); ev.append(value)
        def ub(row, col, value):
            ur.append(row); uc.append(col); uv.append(value)
        for p, options in enumerate(w.options):
            for m in options:
                eq(p, self.nx+m, 1.)
        for node, mode in enumerate(w.node_pattern):
            eq(count_p+node, self.nx+int(mode), -1.)
        for j, c in enumerate(self.columns):
            for n in c.nodes:
                eq(count_p+n, j, 1.)
            eq(count_p+count_b+c.robot, j, 1.)
            ub(c.robot, j, c.cost)
        for r in range(4):
            ub(r, self.z, -1.)
        self.conflicts = []
        # Cache by head geometry; many distinct full routes share a head.
        cache = {}
        for i, a in enumerate(self.columns):
            for j in range(i+1, self.nx):
                b = self.columns[j]
                if a.robot == b.robot or not a.nodes or not b.nodes:
                    continue
                key = (a.robot, a.nodes[0], a.directions[0],
                       b.robot, b.nodes[0], b.directions[0])
                if key not in cache:
                    cache[key] = heads_compatible(w, a, b)
                if not cache[key]:
                    row = 4+len(self.conflicts)
                    self.conflicts.append((i, j))
                    ub(row, i, 1.); ub(row, j, 1.)
        self.aeq = coo_matrix((ev, (er, ec)), shape=(count_p+count_b+4, variables)).tocsr()
        self.beq = np.r_[np.ones(count_p), np.zeros(count_b), np.ones(4)]
        self.aub = coo_matrix((uv, (ur, uc)), shape=(4+len(self.conflicts), variables)).tocsr()
        self.bub = np.r_[np.zeros(4), np.ones(len(self.conflicts))]
        self.objective = np.zeros(variables); self.objective[self.z] = 1.

    def solve_lp(self):
        result = linprog(self.objective, A_ub=self.aub, b_ub=self.bub,
            A_eq=self.aeq, b_eq=self.beq, bounds=(0., None), method='highs')
        if not result.success:
            raise RuntimeError(f'restricted master LP: {result.message}')
        p, b = len(self.w.parents), len(self.w.ids)
        dual = DualPrices(np.array(result.eqlin.marginals[:p]),
            np.array(result.eqlin.marginals[p:p+b]),
            np.array(result.eqlin.marginals[p+b:]), -np.array(result.ineqlin.marginals[:4]))
        dual.existing = {c.key: float(rc) for c, rc in zip(self.columns, self.existing_reduced_costs(result))}
        return result, dual

    def existing_reduced_costs(self, lp):
        return (-self.aeq.T@lp.eqlin.marginals-self.aub.T@lp.ineqlin.marginals)[:self.nx]

    def solve_integer(self, seconds, excluded=()):
        aeq, beq, aub, bub = self.aeq, self.beq, self.aub, self.bub
        extra = []
        for bundle in excluded:
            indices = [i for i, c in enumerate(self.columns) if c.key in {d.key for d in bundle.columns}]
            if len(indices) == 4:
                row = np.zeros(self.z+1); row[indices] = 1.
                extra.append(row)
        if extra:
            aub = vstack((aub, np.array(extra)), format='csr')
            bub = np.r_[bub, np.full(len(extra), 3.)]
        integrality = np.ones(self.z+1, np.int32); integrality[self.z] = 0
        upper = np.ones(self.z+1); upper[self.z] = np.inf
        result = milp(self.objective, integrality=integrality,
            bounds=Bounds(np.zeros(self.z+1), upper),
            constraints=(LinearConstraint(aeq, beq, beq), LinearConstraint(aub, -np.inf, bub)),
            options=dict(time_limit=seconds, mip_rel_gap=0.))
        if result.x is None:
            return result, None
        discrete = result.x[:self.z]
        if np.max(np.abs(discrete-np.rint(discrete)), initial=0.) > 1.e-5:
            return result, None
        picked = [c for c, x in zip(self.columns, discrete[:self.nx]) if x > .5]
        if len(picked) != 4 or sorted(c.robot for c in picked) != list(range(4)):
            raise RuntimeError('MILP did not select exactly one route per robot')
        selected = np.array([next(m for m in options if discrete[self.nx+m] > .5)
            for options in self.w.options], np.int64)
        picked.sort(key=lambda c: c.robot)
        bundle = make_bundle(self.w, [c.nodes for c in picked], [c.directions for c in picked], selected)
        return result, bundle


FEATURE_DIM = 30


class RoutePricingGNN(nn.Module):
    """Reuse VNext's frozen two-layer geometry encoder; train one pricing head."""
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder or VNextGNN()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        hidden = self.encoder.layer2.bias.numel()
        self.head = nn.Sequential(nn.Linear(4*hidden+FEATURE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, rows):
        return self.head(rows).squeeze(-1)


def load_pricer(checkpoint, seed=20261110):
    torch.manual_seed(seed)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    encoder = VNextGNN()
    state = {k: v for k, v in saved['state_dict'].items()
             if k.startswith(('layer1.', 'layer2.', 'geometry_skip.'))}
    required = {k for k in encoder.state_dict() if k.startswith(('layer1.', 'layer2.', 'geometry_skip.'))}
    if set(state) != required:
        raise ValueError('checkpoint lacks a complete VNext geometry encoder')
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return RoutePricingGNN(encoder)


class PricingContext:
    def __init__(self, w, policy):
        self.w, self.policy = w, policy
        self.scale = max(1., sum(min(sum(w.processing[n] for n in w.pattern_nodes[m] if n >= 0)
            for m in options) for options in w.options)/4.)
        self.minimum_parent_work = np.array([min(sum(w.processing[n] for n in w.pattern_nodes[m] if n >= 0)
            for m in options) for options in w.options])
        self.mode_rows = np.array([[float(w.patterns[int(w.node_pattern[n])].kind.value == k)
            for k in ('WHOLE', 'Y_SPLIT', 'X_SPLIT')] for n in range(len(w.ids))])
        self.embeddings = self.node_map = self.parent_embeddings = None
        if policy is not None:
            space = complete_parent_space(w.parents, w.config)
            self.graph = space['graph']
            with torch.no_grad():
                self.embeddings = policy.encoder.encode_static(self.graph).detach()
            self.node_map = torch.tensor([space['indices'][w.patterns[int(w.node_pattern[n])], w.ids[n]]
                for n in range(len(w.ids))], dtype=torch.long)
            self.parent_embeddings = torch.stack([self.embeddings[self.node_map[
                torch.tensor(np.flatnonzero(w.parent_index == p))]].mean(0) for p in range(len(w.parents))])

    def rows(self, robot, nodes, bits, routes, vectors, costs, selected, dual):
        w, scale = self.w, self.scale
        ns, ds = np.asarray(nodes, np.int64), np.asarray(bits, np.int64)
        last = routes[robot][-1] if routes[robot] else None
        ld = vectors[robot][-1] if routes[robot] else 0
        transfer = np.zeros(len(ns)) if last is None else w.travel[last, ns, ld, ds]
        increment = w.processing[ns]+transfer
        immediate = dual.load[robot]*increment-dual.coverage[ns]
        if self.policy is None:
            return None, immediate
        endpoint = w.points[last, 1-ld] if last is not None else np.zeros(2)
        remaining = self.minimum_parent_work[selected < 0].sum()
        count = len(ns)
        constant = lambda value: np.full(count, value)
        features = np.column_stack((constant(dual.load[robot]), dual.coverage[ns]/scale,
            constant(dual.robot[robot]/scale), increment/scale, immediate/scale,
            constant(costs[robot]/scale), np.broadcast_to(costs/scale, (count, 4)),
            constant(sum(w.processing[list(routes[robot])])/scale), constant(remaining/(4*scale)),
            np.ones(count), ds, w.points[ns, ds]/np.array([20., 12.]),
            w.points[ns, 1-ds]/np.array([20., 12.]),
            np.broadcast_to(endpoint/np.array([20., 12.]), (count, 2)), constant(max(costs)/scale),
            self.mode_rows[ns], constant(sum(len(row) for row in routes)/max(1, len(w.parents))),
            w.eligible[ns], selected[w.parent_index[ns]] >= 0))
        numeric = torch.tensor(features, dtype=torch.float32).reshape(-1, FEATURE_DIM)
        h = self.embeddings
        robot_h = h[len(self.graph.blocks)+robot].expand(count, -1)
        last_h = robot_h if last is None else h[self.node_map[last]].expand(count, -1)
        return torch.cat((h[self.node_map[ns]], robot_h, last_h,
            self.parent_embeddings[w.parent_index[ns]], numeric), dim=1), immediate


def append_options(w, r, routes, bits, selected, used):
    nodes, directions = [], []
    for n in np.flatnonzero(w.eligible[:, r]):
        n, parent, mode = int(n), int(w.parent_index[n]), int(w.node_pattern[n])
        if n in used or (selected[parent] >= 0 and selected[parent] != mode):
            continue
        if routes[r] and w.parent_index[routes[r][-1]] == parent:
            continue
        for d in (0, 1):
            if not routes[r]:
                rr = [np.array(row[:1], np.int64) for row in routes]
                dd = [np.array(row[:1], np.int64) for row in bits]
                rr[r], dd[r] = np.array([n], np.int64), np.array([d], np.int64)
                if not _heads_legal(rr, dd, w.points, w.parameters):
                    continue
            nodes.append(n); directions.append(d)
    return np.array(nodes, np.int64), np.array(directions, np.int64)


def _append(w, r, n, d, routes, bits, selected, used, costs):
    costs[r] += w.processing[n]
    if routes[r]:
        costs[r] += w.travel[routes[r][-1], n, bits[r][-1], d]
    routes[r].append(int(n)); bits[r].append(int(d)); used.add(int(n))
    selected[w.parent_index[n]] = w.node_pattern[n]


def construct_bundle(context, dual, initial, variant, *, counts=None):
    """Append-only construction from empty routes, never a neighborhood move.

Two of the four proposals retain only the initial head states; two choose new
heads subject to the same predicate. Mode commitment occurs on the first child.
Complementary routes are constructed together so new task sets/modes can be
covered exactly by the integer master, rather than stranded in a route pool.
"""
    w = context.w
    routes, bits = [[] for _ in range(4)], [[] for _ in range(4)]
    selected, used, costs = np.full(len(w.parents), -1, np.int64), set(), np.zeros(4)
    if variant < 2:
        for c in initial.columns:
            if c.nodes:
                _append(w, c.robot, c.nodes[0], c.directions[0], routes, bits, selected, used, costs)
    steps = 0
    while True:
        expected = sum(sum(n >= 0 for n in w.pattern_nodes[m]) for m in selected if m >= 0)
        if np.all(selected >= 0) and len(used) == expected:
            break
        options = []
        for r in sorted(range(4), key=lambda r: (costs[r], r)):
            ns, ds = append_options(w, r, routes, bits, selected, used)
            if len(ns):
                options = (r, ns, ds)
                break
        if not options:
            return None
        r, ns, ds = options
        rows, greedy = context.rows(r, ns, ds, routes, bits, costs, selected, dual)
        if context.policy is None:
            scores = greedy
        else:
            with torch.no_grad():
                scores = context.policy(rows).numpy()
            if counts is not None:
                counts['gnn_score_batches'] += 1
                counts['gnn_scored_actions'] += len(ns)
                chosen_gnn = min(range(len(ns)), key=lambda i: (scores[i], greedy[i], int(ns[i]), int(ds[i])))
                chosen_greedy = min(range(len(ns)), key=lambda i: (greedy[i], int(ns[i]), int(ds[i])))
                counts['gnn_top_choice_differences'] += int(chosen_gnn != chosen_greedy)
        order = sorted(range(len(ns)), key=lambda i: (float(scores[i]), float(greedy[i]), int(ns[i]), int(ds[i])))
        # Identical deterministic diversity policy in both arms. No new budget
        # is allocated after observing pilot quality.
        rank = int(variant % 2 == 1 and steps % 7 == 0 and len(order) > 1)
        j = order[rank]
        _append(w, r, int(ns[j]), int(ds[j]), routes, bits, selected, used, costs)
        steps += 1
    return make_bundle(w, routes, bits, selected)


def _training_seed(w):
    patterns = tuple(w.patterns[m] for m in w.initial_selected)
    solution = _construct_rail_monotone_balanced_bootstrap(w.parents, patterns, w.config)
    routes = w.encode(solution)
    bits = [_greedy_route_directions(row, w.travel) for row in routes]
    routes, bits = _repair_oriented_first(routes, bits, w.points, w.travel, w.parameters, len(w.ids))
    return make_bundle(w, routes, bits, w.initial_selected)


def _completion_label(context, dual, robot, prefix, prefix_bits, selected, first, target):
    """Complete a single pricing route; label its actual full reduced cost."""
    w = context.w
    routes, vectors = [[] for _ in range(4)], [[] for _ in range(4)]
    routes[robot], vectors[robot] = list(prefix), list(prefix_bits)
    selected, used = selected.copy(), set(prefix)
    costs = np.zeros(4)
    costs[robot] = _oriented_route_cost(np.array(prefix, np.int64), np.array(prefix_bits, np.int64), w.travel, w.processing)
    _append(w, robot, *first, routes, vectors, selected, used, costs)
    for _ in range(2*len(w.parents)):
        ns, ds = append_options(w, robot, routes, vectors, selected, used)
        if not len(ns) or costs[robot] >= target:
            break
        _, rc = context.rows(robot, ns, ds, routes, vectors, costs, selected, dual)
        j = min(range(len(ns)), key=lambda i: (rc[i], int(ns[i]), int(ds[i])))
        if rc[j] >= -1.e-8:
            break
        _append(w, robot, int(ns[j]), int(ds[j]), routes, vectors, selected, used, costs)
    column = make_column(w, robot, routes[robot], vectors[robot])
    return dual.reduced_cost(column)/context.scale


def fit_pricing_head(policy, training_layouts, *, epochs=6, progress_callback=None):
    """TRAIN-only LP/completion labels; zero native/NATIVE/HGA/WAG evaluations."""
    started = time.perf_counter()
    groups, labels, lp_calls, completions = [], [], 0, 0
    for name, parents in training_layouts:
        w = _HGAWorkspace(parents, ScientificConfig(), FORMAL_SCOPE_V2)
        initial = _training_seed(w)
        pools = [[c] for c in initial.columns]
        context, greedy = PricingContext(w, policy), PricingContext(w, None)
        for iteration in range(3):
            _, dual = RestrictedMaster(w, pools).solve_lp(); lp_calls += 1
            for robot in range(4):
                original = initial.columns[robot]
                for fraction in (0., .5):
                    k = int(fraction*len(original.nodes))
                    prefix, vector = original.nodes[:k], original.directions[:k]
                    routes, bits = [[] for _ in range(4)], [[] for _ in range(4)]
                    routes[robot], bits[robot] = list(prefix), list(vector)
                    selected = np.full(len(w.parents), -1, np.int64)
                    for n in prefix:
                        selected[w.parent_index[n]] = w.node_pattern[n]
                    costs = np.zeros(4)
                    costs[robot] = _oriented_route_cost(np.array(prefix, np.int64), np.array(vector, np.int64), w.travel, w.processing)
                    ns, ds = append_options(w, robot, routes, bits, selected, set(prefix))
                    if len(ns) < 2:
                        continue
                    x, rc = context.rows(robot, ns, ds, routes, bits, costs, selected, dual)
                    ranking = sorted(range(len(ns)), key=lambda i: (rc[i], int(ns[i]), int(ds[i])))
                    # Four alternatives from the same state: best, two near
                    # competitors and a separated alternative, not outcome-
                    # selected samples or NATIVE counterfactuals.
                    chosen = sorted(set(ranking[i] for i in (0, 1, min(5, len(ranking)-1), len(ranking)//2)))
                    ys = [_completion_label(greedy, dual, robot, prefix, vector, selected,
                        (int(ns[i]), int(ds[i])), max(original.cost, context.scale)) for i in chosen]
                    groups.append(x[chosen].detach()); labels.append(torch.tensor(ys, dtype=torch.float32))
                    completions += len(chosen)
            for variant in range(4):
                proposal = construct_bundle(greedy, dual, initial, variant)
                if proposal is not None:
                    for c in proposal.columns:
                        if c.key not in {old.key for old in pools[c.robot]}:
                            pools[c.robot].append(c)
        if progress_callback:
            progress_callback(dict(phase='PRICING_LABELS', layout_id=name, groups=len(groups), route_labels=completions))
    if not groups:
        raise RuntimeError('no route pricing ranking groups')
    optimizer = torch.optim.Adam(policy.head.parameters(), lr=.003)
    losses, gradient_norm = [], 0.
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.zeros(())
        for x, target in zip(groups, labels):
            prediction = policy(x)
            loss = loss+F.smooth_l1_loss(prediction, target)
            a, b = torch.triu_indices(len(target), len(target), 1)
            meaningful = torch.abs(target[a]-target[b]) > 1.e-7
            if meaningful.any():
                sign = torch.sign(target[b[meaningful]]-target[a[meaningful]])
                gap = prediction[b[meaningful]]-prediction[a[meaningful]]
                loss = loss+F.softplus(-sign*gap).mean()
        loss = loss/len(groups)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum((p.grad.square().sum() for p in policy.head.parameters() if p.grad is not None))))
        optimizer.step(); losses.append(float(loss.detach()))
    wins = pairs = 0
    with torch.no_grad():
        for x, target in zip(groups, labels):
            prediction = policy(x)
            a, b = torch.triu_indices(len(target), len(target), 1)
            valid = torch.abs(target[a]-target[b]) > 1.e-7
            pairs += int(valid.sum())
            wins += int((((prediction[a]-prediction[b])*(target[a]-target[b]) > 0)&valid).sum())
    policy.eval()
    return dict(layouts=[name for name, _ in training_layouts], epochs=epochs, groups=len(groups),
        complete_route_labels=completions, lp_calls=lp_calls, coordination_calls=0, native_calls=0,
        loss=losses, final_head_gradient_norm=gradient_norm, training_rank_pairs=pairs,
        training_rank_accuracy=wins/pairs if pairs else None, runtime_s=time.perf_counter()-started,
        target='full_route_reduced_cost_from_current_LP', frozen_encoder=True)


@dataclass
class RCGResult:
    status: str
    cmax: float | None
    solution: object
    directions: tuple | None
    schedule: object
    runtime: float
    termination_reason: str
    accounting: dict
    progress: list = field(default_factory=list)
    final_candidates: list = field(default_factory=list)


def optimize_route_columns(parents, initial, *, policy=None, settings=None,
                           initialization_s=None, progress_callback=None):
    """Both arms use the SAME in-run initial result and identical master/budgets.

    Each arm's full cost includes that shared initialization once. Actual batch
    cost counts it only once. The cached initial NATIVE object is never reevaluated.
    """
    settings = settings or RCGConfig()
    started = time.perf_counter()
    init_s = initial.runtime if initialization_s is None else initialization_s
    a = dict(initialization_s=init_s, initial_cmax_s=initial.cmax, rounds=0,
        lp_calls=0, milp_calls=0, master_s=0., column_generation_s=0.,
        proposed_columns=0, added_columns=0, negative_columns=0, companion_columns=0,
        construction_failures=0, duplicate_columns=0, final_coordination_calls=0, final_coordination_s=0.,
        initial_coordination_calls=initial.accounting['coordination_calls'], coordination_calls=initial.accounting['coordination_calls'],
        native_calls=initial.accounting['native_calls'], search_complete=False,
        gnn_score_batches=0, gnn_scored_actions=0, gnn_top_choice_differences=0)
    if initial.status != 'FEASIBLE' or initial.schedule is None:
        return RCGResult(initial.status, initial.cmax, initial.solution, initial.directions,
            initial.schedule, init_s+time.perf_counter()-started, 'INITIALIZATION_FAILED', a)
    w = _HGAWorkspace(parents, ScientificConfig(), FORMAL_SCOPE_V2)
    original = bundle_from_solution(w, initial.solution, initial.directions)
    context = PricingContext(w, policy)
    pools = [[c] for c in original.columns]
    candidates, progress, final_rows = {original.key: original}, [], []
    stale, termination = 0, 'ROUND_LIMIT'
    a['initial_route_periods_s'] = [c.cost for c in original.columns]
    a['initial_pool'] = [dict(robot=c.robot, nodes=list(c.nodes), directions=list(c.directions), cost_s=c.cost)
        for c in original.columns]
    for iteration in range(1, settings.max_rounds+1):
        a['rounds'] = iteration
        t = time.perf_counter(); master = RestrictedMaster(w, pools)
        lp, dual = master.solve_lp(); a['lp_calls'] += 1
        a['master_s'] += time.perf_counter()-t
        added, negative, companion, proposal_rc = 0, 0, 0, []
        t = time.perf_counter()
        for variant in range(settings.columns_per_robot_round):
            proposal = construct_bundle(context, dual, original, variant, counts=a)
            if proposal is None:
                a['construction_failures'] += 1
                continue
            a['proposed_columns'] += 4
            new = [c for c in proposal.columns if c.key not in {d.key for d in pools[c.robot]}]
            a['duplicate_columns'] += 4-len(new)
            prices = [dual.reduced_cost(c) for c in new]
            proposal_rc.extend(prices)
            if not any(value < -1.e-7 for value in prices):
                continue
            if any(len(pools[c.robot]) >= settings.pool_per_robot for c in new):
                continue
            # A negative column's complementary routes are needed to cover its
            # new task set/mode. These are explicitly counted as companions,
            # never falsely described as individually negative reduced cost.
            for c, rc in zip(new, prices):
                pools[c.robot].append(c); added += 1
                negative += int(rc < -1.e-7); companion += int(rc >= -1.e-7)
        a['column_generation_s'] += time.perf_counter()-t
        a['added_columns'] += added; a['negative_columns'] += negative; a['companion_columns'] += companion
        stale = stale+1 if added == 0 else 0
        mip_status, period = None, None
        if added:
            t = time.perf_counter(); master = RestrictedMaster(w, pools)
            mip, item = master.solve_integer(settings.mip_seconds); a['milp_calls'] += 1
            a['master_s'] += time.perf_counter()-t
            mip_status = int(mip.status)
            if item is not None:
                candidates[item.key] = item; period = item.period
        row = dict(iteration=iteration, lp_objective_s=float(lp.fun), load_duals=dual.load.tolist(),
            generated_columns=a['proposed_columns'], added_columns=added, negative_columns=negative,
            companion_columns=companion, pool_sizes=[len(pool) for pool in pools],
            minimum_new_reduced_cost_s=min(proposal_rc) if proposal_rc else None,
            mip_status=mip_status, integer_route_period_s=period,
            elapsed_s=init_s+time.perf_counter()-started)
        progress.append(row)
        if progress_callback:
            progress_callback(dict(phase='GNN_RCG' if policy is not None else 'GREEDY_RCG', **row))
        if stale >= settings.empty_rounds:
            termination = 'NO_NEW_VALUABLE_COLUMNS'; break
    # Obtain a distinct second complete route combination with the same master.
    # Excluding columns of one bundle changes no geometry or scheduling model.
    best = min(candidates.values(), key=lambda b: (b.period, b.key))
    if a['added_columns'] and settings.final_coordination_budget > 1:
        t = time.perf_counter(); master = RestrictedMaster(w, pools)
        mip, item = master.solve_integer(settings.mip_seconds, excluded=(best,)); a['milp_calls'] += 1
        a['master_s'] += time.perf_counter()-t
        if item is not None:
            candidates[item.key] = item
    selected = sorted((b for b in candidates.values() if b.key != original.key and b.period < initial.cmax-1.e-7),
        key=lambda b: (b.period, b.key))[:settings.final_coordination_budget]
    solution, directions, schedule, returned = initial.solution, initial.directions, initial.schedule, original
    for item in selected:
        trial_solution, trial_bits = export_bundle(w, item)
        t = time.perf_counter()
        trial = native_schedule(trial_solution, w.config, scope=FORMAL_SCOPE_V2,
            orientations=dict(enumerate(trial_bits)))
        elapsed = time.perf_counter()-t
        a['final_coordination_calls'] += 1; a['coordination_calls'] += 1; a['final_coordination_s'] += elapsed
        final_rows.append(dict(status=trial.status.value, cmax_s=trial.cmax if trial.feasible else None,
            route_period_s=item.period, coordination_s=elapsed, solution=solution_record(trial_solution, trial_bits),
            diagnostic=' | '.join(trial.diagnostics[:2])))
        if trial.feasible and trial.cmax < schedule.cmax-1.e-7:
            if trial.directions != trial_bits:
                raise RuntimeError('NATIVE changed explicit column directions')
            solution, directions, schedule, returned = trial_solution, trial_bits, trial, item
    rr = [np.array(c.nodes, np.int64) for c in returned.columns]
    dd = [np.array(c.directions, np.int64) for c in returned.columns]
    costs, loads = _route_statistics(rr, dd, w.travel, w.processing)
    a.update(search_complete=True, pool_sizes=[len(pool) for pool in pools],
        final_route_periods_s=costs.tolist(), processing_s=loads.tolist(), move_s=float(sum(costs-loads)),
        wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
        returned_initial=schedule is initial.schedule, best_master_period_s=best.period,
        returned_parent_mode_changes=[w.parents[p].parent_id for p in range(len(w.parents))
            if original.selected[p] != returned.selected[p]],
        changes=geometric_changes(w, original, returned), unique_integer_candidates=len(candidates))
    return RCGResult('FEASIBLE', schedule.cmax, solution, directions, schedule,
        init_s+time.perf_counter()-started, termination, a, progress, final_rows)


def geometric_changes(w, before, after):
    # Reuse the project's geometric-overlap transfer definition even when modes
    # change child IDs/cuts. Counting equal strings would miss real transfers.
    from types import SimpleNamespace
    from mrta_search.lns import actual_parent_transfers
    old, old_bits = export_bundle(w, before); new, new_bits = export_bundle(w, after)
    return actual_parent_transfers(SimpleNamespace(solution=old), SimpleNamespace(solution=new), w.config)


def solution_record(solution, directions):
    if solution is None:
        return None
    return dict(patterns=[dict(parent_id=p.parent_id, kind=p.kind.value, t=p.t,
        point_id=p.point_id, rail=p.rail.value if p.rail else None) for p in solution.patterns],
        routes=[list(r.block_ids) for r in solution.routes], directions=directions)
