"""Supervised GAT ranking and independent, fixed-task bottleneck VNS.

Only numeric geometry, insertion, merging and M1--M6 descriptors are shared
with the baselines. No baseline solver, population or VND is called here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mrta_baselines.hga import (
    _HGAWorkspace, _insert_fixed_oriented, _materialize_directions,
    _materialize_segments, _merge_fixed_arrays, _move_segments,
    _oriented_route_cost, _put_oriented, _repair_oriented_first,
)
from mrta_baselines.wag_vns import _priority_arrays, _schedule_from_arrays
from mrta_reference.fixed_taskset import FixedTaskset
from mrta_reference.model import ScientificConfig, ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2


@dataclass(frozen=True)
class BRConfig:
    alpha: int = 8
    anchors: int = 12
    candidates: int = 40
    top_k: int = 3
    max_rounds: int = 200
    perturbations: int = 3
    initial_attempts: int = 10
    time_limit: float = 180.0

    def __post_init__(self):
        if min(self.alpha, self.anchors, self.candidates, self.top_k,
               self.max_rounds, self.initial_attempts) < 1:
            raise ValueError('search budgets must be positive')
        if self.perturbations < 0 or not 0 < self.time_limit < math.inf:
            raise ValueError('invalid perturbation or time budget')


@dataclass
class State:
    routes: list
    bits: list
    value: float
    completion: np.ndarray
    costs: np.ndarray
    loads: np.ndarray
    operations: list

    @property
    def feasible(self):
        return math.isfinite(self.value)

    @property
    def status(self):
        if math.isnan(self.value):
            return ScheduleStatus.NUMERIC_FAILURE
        return ScheduleStatus.FEASIBLE if self.feasible else ScheduleStatus.DEADLOCK

    @property
    def bottleneck(self):
        return int(np.argmax(self.completion if self.feasible else self.costs))


@dataclass
class Action:
    move: int
    u: int
    v: int
    ru: int
    rv: int
    routes: list
    bits: list
    costs: np.ndarray
    loads: np.ndarray
    features: np.ndarray


@dataclass
class LocalGraph:
    nodes: torch.Tensor
    edges: torch.Tensor
    relations: torch.Tensor
    edge_attr: torch.Tensor
    action_nodes: torch.Tensor
    action_features: torch.Tensor


# Each column is a physical quantity or a one-hot categorical observation.
NODE_FEATURES = (
    'is_job', 'is_robot', 'start_x', 'start_y', 'end_x', 'end_y',
    'length', 'route_position', 'direction',
    'robot_id_0', 'robot_id_1', 'robot_id_2', 'robot_id_3',
    'robot_processing_load', 'robot_completion', 'robot_empty_time',
)
EDGE_FEATURES = ('can_execute', 'is_current_owner', 'directed_empty_time')
ACTION_FEATURES = (
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
    'source_load', 'target_load', 'source_completion', 'target_completion',
    'source_processing_delta', 'target_processing_delta',
    'source_empty_delta', 'target_empty_delta', 'route_peak_delta',
    'source_route_delta', 'target_route_delta', 'source_job_fraction',
)


class SparseGATLayer(nn.Module):
    """Vectorized incoming-edge attention; no dynamic-split output heads."""

    def __init__(self, input_dim, hidden=64, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden dimension must be divisible by heads')
        self.heads, self.width = heads, hidden // heads
        self.linear = nn.Linear(input_dim, hidden, bias=False)
        self.source = nn.Parameter(torch.empty(heads, self.width))
        self.destination = nn.Parameter(torch.empty(heads, self.width))
        # self, job-robot, parent adjacency, successor, predecessor, spatial
        self.relation = nn.Embedding(6, heads)
        self.edge_bias = nn.Linear(len(EDGE_FEATURES), heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.destination)
        nn.init.zeros_(self.relation.weight)

    def forward(self, x, edges, relations, edge_attr):
        h = self.linear(x).reshape(len(x), self.heads, self.width)
        source, destination = edges
        logits = F.leaky_relu(
            (h[source] * self.source).sum(-1)
            + (h[destination] * self.destination).sum(-1)
            + self.relation(relations) + self.edge_bias(edge_attr), .2)
        index = destination[:, None].expand(-1, self.heads)
        maximum = logits.new_full((len(x), self.heads), -torch.inf)
        maximum.scatter_reduce_(0, index, logits.detach(), reduce='amax', include_self=True)
        weights = (logits - maximum[destination]).exp()
        denominator = weights.new_zeros((len(x), self.heads))
        denominator.scatter_add_(0, index, weights)
        weights = weights / denominator[destination].clamp_min(1.e-12)
        output = h.new_zeros(h.shape)
        output.index_add_(0, destination, h[source] * weights[:, :, None])
        return output.reshape(len(x), -1) + self.bias


class BottleneckGAT(nn.Module):
    """One shared two-layer encoder and one scalar action ranking head."""

    def __init__(self, hidden=64, heads=4, *, use_edge_features=True):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.layer1 = SparseGATLayer(len(NODE_FEATURES), hidden, heads)
        self.layer2 = SparseGATLayer(hidden, hidden, heads)
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden + len(ACTION_FEATURES), hidden), nn.ELU(),
            nn.Linear(hidden, 1))

    def encode(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_features else torch.zeros_like(graph.edge_attr)
        h = F.elu(self.layer1(graph.nodes, graph.edges, graph.relations, edge_attr))
        return F.elu(self.layer2(h, graph.edges, graph.relations, edge_attr))

    def score(self, graph, embedding):
        involved = embedding[graph.action_nodes].flatten(1)
        return self.scorer(torch.cat((involved, graph.action_features), dim=1)).squeeze(-1)

    def forward(self, graph):
        return self.score(graph, self.encode(graph))


class ActionMLP(nn.Module):
    """The requested control: rank only the same 18 action features."""

    def __init__(self, hidden=64):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(len(ACTION_FEATURES), hidden), nn.ELU(),
                                    nn.Linear(hidden, 1))

    def forward(self, graph):
        return self.scorer(graph.action_features).squeeze(-1)


MODEL_VARIANTS = ('full_gnn', 'no_edge_gnn', 'action_mlp')


def make_ranker(variant='full_gnn'):
    if variant == 'action_mlp':
        return ActionMLP()
    if variant in ('full_gnn', 'no_edge_gnn'):
        return BottleneckGAT(use_edge_features=variant == 'full_gnn')
    raise ValueError(f'unknown ranking model: {variant}')


class FixedWorkspace:
    """Reuse static fixed-task numeric tables, never HGA's search routines."""

    def __init__(self, taskset, config=ScientificConfig()):
        if not isinstance(taskset, FixedTaskset):
            raise TypeError('GNN-VNS-BR requires a FixedTaskset with immutable atom boundaries')
        self.taskset, self.config = taskset, config
        self.numeric = _HGAWorkspace(taskset.parents, config, FORMAL_SCOPE_V2,
                                     patterns=taskset.patterns)
        w = self.numeric
        self.index = {identifier: i for i, identifier in enumerate(w.ids)}
        self.scale = max(1., sum(config.process_time(p.length) for p in taskset.parents) / 4.)
        self.lengths = np.linalg.norm(w.points[:, 1] - w.points[:, 0], axis=1)
        self.centers = w.points.mean(axis=1)
        self.evaluations = 0

    def costs(self, routes, bits):
        w = self.numeric
        costs = np.array([_oriented_route_cost(r, d, w.travel, w.processing)
                          for r, d in zip(routes, bits)])
        loads = np.array([w.processing[r].sum() for r in routes])
        return costs, loads

    def evaluate(self, routes, bits):
        w = self.numeric
        value, vectors, operations, completion = _priority_arrays(
            routes, w.travel, w.processing, w.points, w.parameters, bits)
        self.evaluations += 1
        costs, loads = self.costs(routes, vectors)
        return State(routes, vectors, float(value), completion, costs, loads, operations)

    def restore(self, payload):
        """Read a completed TRAIN teacher without rerunning its algorithm."""
        routes = [np.array([self.index[i] for i in row], np.int64) for row in payload['routes']]
        bits = [np.array(row, np.int64) for row in payload['directions']]
        self.validate(routes, bits)
        return self.evaluate(routes, bits)

    def solution(self, state):
        return self.numeric.solution(state.routes, self.numeric.initial_selected)

    def schedule_result(self, state):
        """Package the already computed arrays; do not invoke the kernel again."""
        return _schedule_from_arrays(state.value, state.bits, state.operations, state.completion,
                                     self.numeric.ids, 'NATIVE_PRIORITY_FOUR_ROBOTS_EXPLICIT_DIRECTIONS')

    def validate(self, routes, bits):
        w = self.numeric
        if len(routes) != 4 or len(bits) != 4:
            raise ValueError('exactly four routes and direction vectors are required')
        coverage = np.zeros(len(w.atomic_nodes), np.int64)
        for r in range(4):
            if len(routes[r]) != len(bits[r]) or np.any((bits[r] != 0) & (bits[r] != 1)):
                raise ValueError('invalid direction genes')
            if np.any(~w.eligible[routes[r], r]):
                raise ValueError('robot reachability violation')
            for node in routes[r]:
                coverage[w.fixed_starts[node]:w.fixed_stops[node]] += 1
        if np.any(coverage != 1):
            raise ValueError('fixed atoms must be covered exactly once')
        merged, ds = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
        if any(not np.array_equal(a, b) for a, b in zip(routes, merged)):
            raise ValueError('contiguous owned atoms have not been merged')
        return True

    def initialize(self, rng, settings, deadline):
        """Independent spatial and randomized greedy insertion starts.

        Restarts are initialization attempts, with no population, crossover,
        local VND, baseline result file, or baseline final solution access.
        """
        w = self.numeric
        best = None
        for attempt in range(settings.initial_attempts):
            if best is not None and time.perf_counter() >= deadline:
                break
            routes = [np.empty(0, np.int64) for _ in range(4)]
            bits = [np.empty(0, np.int64) for _ in range(4)]
            if attempt < max(2, settings.initial_attempts // 2):
                # Spatial greedy assignment of atoms; long parent welds can
                # cross the allocation boundary without changing a single cut.
                groups = [[], []]
                rail_load = np.zeros(2)
                for node in sorted(w.atomic_nodes, key=lambda n: (-self.lengths[n], int(n))):
                    eligible = [rail for rail in (0, 1) if w.eligible[node, 2 * rail]]
                    rail = min(eligible, key=lambda r: (rail_load[r], r))
                    groups[rail].append(int(node))
                    rail_load[rail] += w.processing[node]
                for rail, nodes in enumerate(groups):
                    nodes.sort(key=lambda n: (self.centers[n, 0], self.centers[n, 1], n))
                    prefix = np.cumsum([self.lengths[n] for n in nodes])
                    if not len(nodes):
                        continue
                    fraction = .5 if attempt == 0 else .5 + rng.uniform(-.12, .12)
                    boundary = int(np.searchsorted(prefix, prefix[-1] * fraction)) + 1
                    for side, owned in enumerate((nodes[:boundary], nodes[boundary:])):
                        r = 2 * rail + side
                        routes[r] = np.array(owned, np.int64)
                        bits[r] = np.zeros(len(owned), np.int64)
                routes, bits = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
                # Greedy next directed job, starting at the spatial outside.
                for r in range(4):
                    pending = list(map(int, routes[r]))
                    ordered, directions = [], []
                    point = np.array([self.config.workspace_x[r % 2], 9. if r < 2 else 3.])
                    while pending:
                        _, node, direction = min(
                            (float(np.linalg.norm(w.points[n, d] - point)), n, d)
                            for n in pending for d in (0, 1))
                        pending.remove(node)
                        ordered.append(node); directions.append(direction)
                        point = w.points[node, 1 - direction]
                    routes[r] = np.array(ordered, np.int64)
                    bits[r] = np.array(directions, np.int64)
            else:
                pending = list(map(int, w.atomic_nodes))
                rng.shuffle(pending)
                for node in pending:
                    r, position, bit = _insert_fixed_oriented(
                        node, routes, bits, w.eligible, w.travel, w.processing,
                        w.fixed_starts, w.fixed_stops, w.fixed_lookup, True)
                    _put_oriented(routes, bits, node, bit, r, position)
                    routes, bits = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
            routes, bits = _repair_oriented_first(routes, bits, w.points, w.travel,
                                                   w.parameters, len(w.processing))
            state = self.evaluate(routes, bits)
            key = (not state.feasible, state.value if state.feasible else max(state.costs))
            old = ((not best.feasible, best.value if best.feasible else max(best.costs))
                   if best is not None else (True, math.inf))
            if key < old:
                best = state
        return best

    def apply(self, state, move, ru, iu, rv, iv, flip=False, portion=0):
        """M1--M6 on merged jobs, carrying/inverting direction genes jointly."""
        w = self.numeric
        if move not in range(1, 7) or ru not in range(4) or rv not in range(4):
            raise ValueError('invalid neighborhood')
        if not 0 <= iu < len(state.routes[ru]):
            return None
        if not -1 <= iv < len(state.routes[rv]):
            return None
        if move == 4:
            if ru != rv or abs(iu - iv) < 1:
                return None
        elif ru == rv:
            return None
        if move != 1 and iv < 0:
            return None
        if move == 3 and (iu + 1 >= len(state.routes[ru]) or iv + 1 >= len(state.routes[rv])):
            return None
        if move == 6 and iu + 1 == len(state.routes[ru]) and iv + 1 == len(state.routes[rv]):
            return None
        u = int(state.routes[ru][iu])
        v = int(state.routes[rv][iv]) if iv >= 0 else -1
        moved = u
        if portion:
            # M1 may move a fixed end atom out of a merged job. The remaining
            # interval and transferred atom both use the original static cuts;
            # ownership changes first, then maximal owned runs are remerged.
            # No new welding boundary is generated or switched by this move.
            lo, hi = int(w.fixed_starts[u]), int(w.fixed_stops[u])
            if move != 1 or portion not in (-1, 1) or hi - lo < 2:
                return None
            if portion == 1:
                moved, remaining = int(w.fixed_lookup[lo, lo + 1]), int(w.fixed_lookup[lo + 1, hi])
            else:
                moved, remaining = int(w.fixed_lookup[hi - 1, hi]), int(w.fixed_lookup[lo, hi - 1])
            routes, bits = list(state.routes), list(state.bits)
            routes[ru] = routes[ru].copy(); routes[ru][iu] = remaining
            routes[rv] = np.insert(routes[rv], iv + 1, moved).astype(np.int64)
            bits[rv] = np.insert(bits[rv], iv + 1, int(state.bits[ru][iu])).astype(np.int64)
        else:
            a, b = _move_segments(state.routes, move, ru, iu, rv, iv)
            routes, bits = list(state.routes), list(state.bits)
            routes[ru] = _materialize_segments(a, state.routes)
            bits[ru] = _materialize_directions(a, state.bits)
            if ru != rv:
                routes[rv] = _materialize_segments(b, state.routes)
                bits[rv] = _materialize_directions(b, state.bits)
        if flip and move in (1, 2, 3):
            # Orientation changes belong to the relocation/swap itself.
            for r in {ru, rv}:
                bits[r] = bits[r].copy()
                mask = routes[r] == moved
                bits[r][mask] = 1 - bits[r][mask]
        if any(np.any(~w.eligible[routes[r], r]) for r in {ru, rv}):
            return None
        routes, bits = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
        if all(np.array_equal(a, b) for a, b in zip(routes, state.routes)) and all(
                np.array_equal(a, b) for a, b in zip(bits, state.bits)):
            return None
        costs, loads = self.costs(routes, bits)
        feature = np.zeros(len(ACTION_FEATURES), np.float32)
        feature[move - 1] = 1.
        completion = state.completion if state.feasible else state.costs
        source_empty = (costs[ru] - loads[ru]) - (state.costs[ru] - state.loads[ru])
        target_empty = (costs[rv] - loads[rv]) - (state.costs[rv] - state.loads[rv])
        feature[6:17] = np.array([
            state.loads[ru], state.loads[rv], completion[ru], completion[rv],
            loads[ru] - state.loads[ru], loads[rv] - state.loads[rv],
            source_empty, target_empty, max(costs) - max(state.costs),
            costs[ru] - state.costs[ru], costs[rv] - state.costs[rv]]) / self.scale
        feature[17] = w.processing[moved] / max(1., state.loads[ru])
        return Action(move, u, v, ru, rv, routes, bits, costs, loads, feature)


def candidate_actions(workspace, state, settings, rng):
    """Bounded round-robin budgets across neighborhoods, robots and anchors.

    Directed travel orders each robot's local neighbors. A low route-cost
    proxy never removes a proposal before GNN scoring and native evaluation.
    """
    w = workspace.numeric
    heavy = state.bottleneck
    route = state.routes[heavy]
    if not len(route):
        return []
    anchors = {0, len(route) - 1}
    longest = sorted(range(len(route)), key=lambda i: (-w.processing[route[i]], i))
    anchors.update(longest[:max(1, settings.anchors // 3)])
    remaining = [i for i in range(len(route)) if i not in anchors]
    anchors.update(rng.sample(remaining, min(len(remaining), max(0, settings.anchors - len(anchors)))))
    anchors = sorted(anchors)
    # Each target owns one proposal stream per source anchor. Rotating these
    # streams gives every reachable target/anchor a chance before a repeat.
    buckets = [dict() for _ in range(6)]
    def offer(move, rv, iu, iv, flip=False, portion=0):
        buckets[move - 1].setdefault(rv, {}).setdefault(iu, []).append(
            (heavy, iu, rv, iv, flip, portion))

    for ordinal, iu in enumerate(anchors):
        u, du = int(route[iu]), int(state.bits[heavy][iu])
        for rv in range(4):
            target = state.routes[rv]
            if len(target):
                distance = w.travel[u, target, du, state.bits[rv]]
                indices = sorted(range(len(target)), key=lambda j: (float(distance[j]), int(target[j])))
                if rv == heavy:
                    for iv in [j for j in indices if j != iu][:settings.alpha]:
                        offer(4, rv, iu, iv)
                    continue
                # Eligibility is considered per operator below. In particular
                # a movable end atom can have broader eligibility than its job.
                nearest = indices[:settings.alpha]
            else:
                if rv == heavy:
                    continue
                nearest = [-1]
            lo, hi = int(w.fixed_starts[u]), int(w.fixed_stops[u])
            variants = [(False, 0), (True, 0)]
            if hi - lo > 1:
                variants += [(False, 1), (False, -1), (True, 1), (True, -1)]
            shift = (ordinal + rv) % len(variants)
            variants = variants[shift:] + variants[:shift]
            for iv in nearest:
                for flip, portion in variants:
                    moved = u if not portion else int(
                        w.fixed_lookup[lo, lo + 1] if portion == 1 else w.fixed_lookup[hi - 1, hi])
                    if w.eligible[moved, rv]:
                        offer(1, rv, iu, iv, flip, portion)
            if len(target):
                if w.eligible[u, rv]:
                    legal = [j for j in indices if w.eligible[target[j], heavy]]
                    for iv in legal[:settings.alpha]:
                        offer(2, rv, iu, iv)
                if iu + 1 < len(route) and np.all(w.eligible[route[iu:iu + 2], rv]):
                    legal = [j for j in indices if j + 1 < len(target)
                             and np.all(w.eligible[target[j:j + 2], heavy])]
                    for iv in legal[:settings.alpha]:
                        offer(3, rv, iu, iv)
                if np.all(w.eligible[route[iu + 1:], rv]):
                    prefix = np.logical_and.accumulate(w.eligible[target, heavy])
                    suffix = np.logical_and.accumulate(w.eligible[target[::-1], heavy])[::-1]
                    for iv in [j for j in indices if prefix[j]][:settings.alpha]:
                        offer(5, rv, iu, iv)
                    for iv in [j for j in indices if j + 1 == len(target) or suffix[j + 1]][:settings.alpha]:
                        offer(6, rv, iu, iv)
                # Retain insertion before the current first job as well.
                for flip, portion in variants:
                    moved = u if not portion else int(
                        w.fixed_lookup[lo, lo + 1] if portion == 1 else w.fixed_lookup[hi - 1, hi])
                    if w.eligible[moved, rv]:
                        offer(1, rv, iu, -1, flip, portion)

    output = []
    for move, by_target in enumerate(buckets, 1):
        targets = sorted(by_target)
        if not targets:
            continue
        rotation = rng.randrange(len(targets))
        targets = targets[rotation:] + targets[:rotation]
        streams = {rv: {iu: iter(raw) for iu, raw in by_target[rv].items()} for rv in targets}
        anchor_order = {rv: list(by_target[rv]) for rv in targets}
        cursors = {rv: 0 for rv in targets}
        candidates, seen = [], set()
        attempts = 0
        # This cap bounds array construction even when most transfers are
        # illegal. Failed proposals still rotate to the next target and anchor.
        while len(candidates) < settings.candidates and attempts < 8 * settings.candidates:
            advanced = False
            for rv in targets:
                order = anchor_order[rv]
                descriptor = None
                while order:
                    k = cursors[rv] % len(order)
                    iu = order[k]
                    descriptor = next(streams[rv][iu], None)
                    if descriptor is None:
                        order.pop(k)
                    else:
                        cursors[rv] = k + 1
                        break
                if descriptor is None:
                    continue
                advanced = True; attempts += 1
                action = workspace.apply(state, move, *descriptor)
                if action is not None:
                    key = tuple((tuple(r), tuple(d)) for r, d in zip(action.routes, action.bits))
                    if key not in seen:
                        seen.add(key); candidates.append(action)
                if len(candidates) >= settings.candidates or attempts >= 8 * settings.candidates:
                    break
            if not advanced:
                break
        output.extend(candidates)
    return output


def local_graph(workspace, state, actions):
    """The shared 16-node-feature/3-edge-feature training/inference path."""
    w = workspace.numeric
    location = {int(n): (r, i) for r in range(4) for i, n in enumerate(state.routes[r])}
    included = set(map(int, state.routes[state.bottleneck]))
    included.update(a.u for a in actions)
    included.update(a.v for a in actions if a.v >= 0)
    for n in tuple(included):
        r, i = location[n]
        included.update(map(int, state.routes[r][max(0, i - 1):i + 2]))
    nodes = sorted(included)
    index = {n: i for i, n in enumerate(nodes)}
    robot_base = len(nodes)
    x = np.zeros((len(nodes) + 4, len(NODE_FEATURES)), np.float32)
    completion = state.completion if state.feasible else state.costs
    c = workspace.config
    origin = np.array([c.workspace_x[0], c.workspace_y[0]])
    span = np.array([c.workspace_x[1] - c.workspace_x[0], c.workspace_y[1] - c.workspace_y[0]])
    directions = {n: int(state.bits[r][i]) for n, (r, i) in location.items()}
    for n, j in index.items():
        r, i = location[n]
        x[j, 0] = 1.
        x[j, 2:6] = ((w.points[n] - origin) / span).flatten()
        x[j, 6:9] = (workspace.lengths[n] / 24.,
                     i / max(1, len(state.routes[r]) - 1), directions[n])
        # Robot identity and robot state stay zero on every job node.
    for r in range(4):
        x[robot_base + r, 1] = 1.
        x[robot_base + r, 9 + r] = 1.
        x[robot_base + r, 13:16] = np.array([
            state.loads[r], completion[r], state.costs[r] - state.loads[r]]) / workspace.scale

    edges = {(i, i, 0): (0., 0., 0.) for i in range(len(x))}
    def job_edge(a, b, relation):
        travel = float(w.travel[a, b, directions[a], directions[b]]) / workspace.scale
        edges[index[a], index[b], relation] = (0., 0., travel)

    for n, j in index.items():
        r, i = location[n]
        # All four robots, including ineligible robots, have explicit edges.
        for robot in range(4):
            attr = (float(w.eligible[n, robot]), float(r == robot), 0.)
            edges[j, robot_base + robot, 1] = attr
            edges[robot_base + robot, j, 1] = attr
        if i and int(state.routes[r][i - 1]) in index:
            previous = int(state.routes[r][i - 1])
            job_edge(previous, n, 3)  # route successor
            job_edge(n, previous, 4)  # route predecessor, its own directed cost
        # Sparse spatial transitions use the current exit/entry directions.
        nearest = sorted((float(w.travel[n, m, directions[n], directions[m]]), m)
                         for m in nodes if m != n)[:4]
        for _, m in nearest:
            job_edge(n, m, 5)
            job_edge(m, n, 5)  # recompute; the reverse cost is not copied

    by_parent = {}
    for n in nodes:
        by_parent.setdefault(int(w.parent_index[n]), []).append(n)
    for group in by_parent.values():
        group.sort(key=lambda n: int(w.fixed_starts[n]))
        for a, b in zip(group, group[1:]):
            if w.fixed_stops[a] == w.fixed_starts[b]:
                edges[index[a], index[b], 2] = (0., 0., 0.)
                edges[index[b], index[a], 2] = (0., 0., 0.)
    ordered = sorted(edges)
    involved = [[index[a.u], index[a.v] if a.v >= 0 else robot_base + a.rv,
                 robot_base + a.ru, robot_base + a.rv] for a in actions]
    return LocalGraph(
        torch.from_numpy(x),
        torch.tensor([(a, b) for a, b, _ in ordered], dtype=torch.long).t().contiguous(),
        torch.tensor([relation for _, _, relation in ordered], dtype=torch.long),
        torch.tensor([edges[key] for key in ordered], dtype=torch.float32),
        torch.tensor(involved, dtype=torch.long).reshape(-1, 4),
        torch.tensor(np.array([a.features for a in actions]), dtype=torch.float32).reshape(-1, len(ACTION_FEATURES)))


def ranking_pairs(values, rng, limit=256):
    """Finite native Cmax ranks ahead of DEADLOCK; exclude numeric failures."""
    possible = [(i, j) for i in range(len(values)) for j in range(i + 1, len(values))
                if not (math.isnan(values[i]) or math.isnan(values[j]))
                and (math.isfinite(values[i]) or math.isfinite(values[j]))
                and not (math.isfinite(values[i]) and math.isfinite(values[j])
                         and abs(values[i] - values[j]) <= 1.e-7)]
    if len(possible) > limit:
        possible = rng.sample(possible, limit)
    return torch.tensor([(i, j) if values[i] < values[j] else (j, i)
                         for i, j in possible], dtype=torch.long).reshape(-1, 2)


def train_ranker(examples, *, seed=20261126, epochs=24, learning_rate=3.e-4,
                 variant='full_gnn', progress=print):
    """All examples are TRAIN-derived; hold out complete TRAIN layouts."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    layouts = sorted({name for name, _, _ in examples})
    holdout = set(layouts[::5]) if len(layouts) >= 5 else set()
    training = [e for e in examples if e[0] not in holdout]
    held = [e for e in examples if e[0] in holdout]
    if not training:
        raise ValueError('no informative TRAIN ranking examples')
    model = make_ranker(variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); rng.shuffle(training)
        losses = []
        for _, graph, pairs in training:
            scores = model(graph)
            loss = F.softplus(scores[pairs[:, 1]] - scores[pairs[:, 0]]).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            held_losses, correct, total = [], 0, 0
            for _, graph, pairs in held:
                scores = model(graph)
                delta = scores[pairs[:, 0]] - scores[pairs[:, 1]]
                held_losses.append(float(F.softplus(-delta).mean()))
                correct += int((delta > 0).sum()); total += len(delta)
        criterion = float(np.mean(held_losses if held_losses else losses))
        if criterion < best_loss:
            best_loss, best_epoch, best_state = criterion, epoch, copy.deepcopy(model.state_dict())
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                      held_train_loss=criterion, held_train_pair_accuracy=correct / max(1, total))
        history.append(record); progress('BR_EPOCH', dict(model=variant, **record))
    model.load_state_dict(best_state); model.eval()
    return model, dict(variant=variant, selected_epoch=best_epoch, history=history,
                       epochs=epochs, seed=seed, learning_rate=learning_rate,
                       parameters=sum(p.numel() for p in model.parameters()),
                       train_layouts=sorted(set(layouts) - holdout), held_train_layouts=sorted(holdout),
                       feature_names=NODE_FEATURES, action_feature_names=ACTION_FEATURES)


def run_gnn_vns_br(taskset, model, *, seed, settings=BRConfig(), config=ScientificConfig(), progress=None):
    """Independent initialization, bottleneck VNS and bounded guided shaking.

    Returns the best native-feasible state with its cached native timeline.
    The experiment entry point packages these arrays without rescheduling.
    """
    started = time.perf_counter()
    deadline = started + settings.time_limit
    rng = random.Random(seed)
    workspace = FixedWorkspace(taskset, config)
    current = workspace.initialize(rng, settings, deadline)
    if current is None:
        raise RuntimeError('initialization did not produce a complete fixed-task state')
    best = current if current.feasible else None
    initial_value = current.value
    rounds = accepted = shakes = encodings = 0
    generated_by_move = [0] * 6
    evaluated_by_move = [0] * 6
    feasible_by_move = [0] * 6
    accepted_by_move = [0] * 6
    inference_seconds = 0.
    model.eval()
    while rounds < settings.max_rounds and time.perf_counter() < deadline:
        actions = candidate_actions(workspace, current, settings, rng)
        if not actions:
            break
        for action in actions:
            generated_by_move[action.move - 1] += 1
        tick = time.perf_counter()
        graph = local_graph(workspace, current, actions)
        with torch.inference_mode():
            scores = model(graph).cpu().numpy()
        inference_seconds += time.perf_counter() - tick
        encodings += 1
        improved = False
        evaluated = {}
        winner = None
        winner_move = None
        for move in range(1, 7):
            indices = [i for i, a in enumerate(actions) if a.move == move]
            indices.sort(key=lambda i: (-float(scores[i]), i))
            for i in indices[:settings.top_k]:
                if time.perf_counter() >= deadline:
                    break
                action = actions[i]
                candidate = workspace.evaluate(action.routes, action.bits)
                evaluated[i] = candidate
                evaluated_by_move[move - 1] += 1
                feasible_by_move[move - 1] += int(candidate.feasible)
                # Infeasible states are used only to seek a first feasible one.
                # They never replace a feasible current or global incumbent.
                if candidate.feasible and (not current.feasible or candidate.value < current.value - 1.e-7):
                    if winner is None or candidate.value < winner.value:
                        winner = candidate
                        winner_move = move
        if winner is not None:
            # Compare the native Cmax of all six neighborhoods before moving.
            # GNN scores determine only which bounded candidates are evaluated.
            current = winner; accepted += 1; improved = True
            accepted_by_move[winner_move - 1] += 1
            if best is None or current.value < best.value:
                best = current
        rounds += 1
        if progress is not None:
            progress(rounds, best.value if best else None, workspace.evaluations)
        if improved:
            continue
        if shakes >= settings.perturbations:
            break
        shakes += 1
        shaken = None
        # Some fixed eligibility domains have no feasible M5/M6 exchange.
        # Spend the same bounded shaking attempt on a GNN-ranked M1/M2 move
        # in that case, instead of terminating with unused escape budget.
        for families in ((5, 6), (1, 2)):
            large = [i for i, a in enumerate(actions) if a.move in families]
            large.sort(key=lambda i: (-float(scores[i]), i))
            for i in large[:settings.top_k]:
                if time.perf_counter() >= deadline:
                    break
                candidate = evaluated.get(i)
                if candidate is None:
                    candidate = workspace.evaluate(actions[i].routes, actions[i].bits)
                if candidate.feasible:
                    shaken = candidate; break
            if shaken is not None:
                break
        if shaken is None:
            # A bounded fresh greedy state is an initialization fallback only
            # when no native-feasible incumbent has ever been found.
            if best is None:
                current = workspace.initialize(rng, settings, deadline)
                if current is not None and current.feasible:
                    best = current
                continue
            break
        current = shaken
        if best is None or current.value < best.value:
            best = current
    result = best if best is not None else current
    workspace.validate(result.routes, result.bits)
    return workspace, result, dict(
        rounds=rounds, accepted=accepted, perturbations=shakes,
        native_evaluations=workspace.evaluations, graph_encodings=encodings,
        inference_and_graph_s=inference_seconds,
        generated_by_move=generated_by_move, evaluated_by_move=evaluated_by_move,
        feasible_by_move=feasible_by_move, accepted_by_move=accepted_by_move,
        initial_native_cmax_s=initial_value if math.isfinite(initial_value) else None,
        search_runtime_s=time.perf_counter() - started,
        termination_reason='TIME_LIMIT' if time.perf_counter() >= deadline else
                           'ROUND_LIMIT' if rounds >= settings.max_rounds else 'LOCAL_OPTIMUM',
        parameters=asdict(settings))
