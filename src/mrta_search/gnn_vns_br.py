"""Supervised GAT ranking and independent, fixed-task bottleneck VNS.

Only numeric geometry, insertion, merging and M1--M6 descriptors are shared
with the baselines. No baseline solver, population or VND is called here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from mrta_baselines.wag_vns import _home, _numeric_templates, _priority_arrays, _schedule_from_arrays
from mrta_reference.fixed_taskset import FixedTaskset
from mrta_reference.model import ScientificConfig, ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2


@dataclass(frozen=True)
class FeatureConfig:
    # Median sum(process_time(atom.length))/4 over the 3000 scale1000 TRAIN
    # layouts. A training run may fit another TRAIN-only value, saved in its
    # checkpoint and reused unchanged for every graph at inference.
    time_ref: float = 6193.805663528889
    length_ref: float = 24.0
    x_bounds: tuple = (0., 20.)
    y_bounds: tuple = (0., 12.)

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.time_ref, self.length_ref)):
            raise ValueError('feature reference scales must be finite and positive')
        if tuple(self.x_bounds) != (0., 20.) or tuple(self.y_bounds) != (0., 12.):
            raise ValueError('graph coordinates require the fixed 20m by 12m platform')


@dataclass(frozen=True)
class BRConfig:
    alpha: int = 8
    anchors: int = 12
    candidates: int = 40
    top_k: int = 3
    max_rounds: int = 200
    perturbations: int = 3
    initial_attempts: int = 6
    time_limit: float = 180.0
    temperature: float = 1.0

    def __post_init__(self):
        if min(self.alpha, self.anchors, self.candidates, self.top_k,
               self.max_rounds, self.initial_attempts) < 1:
            raise ValueError('search budgets must be positive')
        if self.perturbations < 0 or not 0 < self.time_limit < math.inf:
            raise ValueError('invalid perturbation or time budget')
        if not 0 < self.temperature < math.inf:
            raise ValueError('temperature must be finite and positive')


@dataclass
class State:
    routes: list
    bits: list
    value: float
    completion: np.ndarray
    costs: np.ndarray
    loads: np.ndarray
    operations: list
    evaluation_status: ScheduleStatus | None = None

    @property
    def feasible(self):
        return self.status is ScheduleStatus.FEASIBLE

    @property
    def status(self):
        if self.evaluation_status is not None:
            return self.evaluation_status
        if math.isnan(self.value):
            return ScheduleStatus.NUMERIC_FAILURE
        return ScheduleStatus.FEASIBLE if math.isfinite(self.value) else ScheduleStatus.DEADLOCK

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
    iu: int
    iv: int
    flip: bool
    portion: int
    moved: int

    @property
    def joint(self):
        return self.move, self.ru, self.iu, self.rv, self.iv, self.flip, self.portion


@dataclass
class LocalGraph:
    nodes: torch.Tensor
    edges: torch.Tensor
    relations: torch.Tensor
    edge_attr: torch.Tensor
    action_nodes: torch.Tensor
    action_features: torch.Tensor
    action_members: torch.Tensor
    task_ids: tuple
    parent_ids: tuple
    atom_nodes: tuple
    feedback_valid: torch.Tensor
    normalization: FeatureConfig


# Each column is a physical quantity or a one-hot categorical observation.
NODE_FEATURES = (
    'is_task_node', 'is_robot_node', 'start_x', 'start_y', 'end_x', 'end_y',
    'length', 'u_start', 'u_end', 'is_assigned', 'route_position', 'direction',
)
ROBOT_FEATURES = (
    'is_task_node', 'is_robot_node', 'R1', 'R2', 'R3', 'R4', 'tcp_x', 'tcp_y',
    'processing_load', 'empty_time', 'completion', 'waiting_time',
)
EDGE_RELATIONS = ('self', 'task_robot', 'parent_next', 'parent_prev',
                  'route_next', 'route_prev', 'space_neighbor')
# Relation-specific meanings: task_robot=(eligible, owner, T_start, T_end),
# route=(0, 0, T_empty, 0), space=(T00, T01, T10, T11), others=zeros.
EDGE_FEATURES = ('relation_value_0', 'relation_value_1', 'relation_value_2', 'relation_value_3')
ACTION_FEATURES = (
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
    'source_processing_delta', 'target_processing_delta',
    'source_empty_delta', 'target_empty_delta', 'route_peak_delta',
    'source_job_fraction',
    'flip', 'portion_none', 'portion_start', 'portion_end',
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
        self.relation = nn.Embedding(len(EDGE_RELATIONS), heads)
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

    def __init__(self, hidden=64, heads=4, *, use_edge_features=True, normalization=FeatureConfig()):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.normalization, self.trained = normalization, False
        self.task_projection = nn.Linear(len(NODE_FEATURES), hidden)
        self.robot_projection = nn.Linear(len(ROBOT_FEATURES), hidden)
        self.feedback_embedding = nn.Embedding(2, hidden)
        self.layer1 = SparseGATLayer(hidden, hidden, heads)
        self.layer2 = SparseGATLayer(hidden, hidden, heads)
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden + len(ACTION_FEATURES), hidden), nn.ELU(),
            nn.Linear(hidden, 1))

    def encode(self, graph):
        if graph.normalization != self.normalization:
            raise ValueError('graph/model normalization differs')
        edge_attr = graph.edge_attr if self.use_edge_features else torch.zeros_like(graph.edge_attr)
        x = torch.where(graph.nodes[:, 1:2].bool(), self.robot_projection(graph.nodes),
                        self.task_projection(graph.nodes))
        feedback = torch.zeros_like(x)
        feedback[graph.nodes[:, 1].bool()] = self.feedback_embedding(graph.feedback_valid.long())
        x = x + feedback
        h = F.elu(self.layer1(F.elu(x), graph.edges, graph.relations, edge_attr))
        return F.elu(self.layer2(h, graph.edges, graph.relations, edge_attr))

    def score(self, graph, embedding):
        members, groups = graph.action_members
        involved = embedding.new_zeros((len(graph.action_features) * 4, embedding.shape[1]))
        involved.index_add_(0, groups, embedding[members])
        counts = torch.bincount(groups, minlength=len(involved)).clamp_min(1)
        involved = (involved / counts[:, None]).reshape(len(graph.action_features), 4 * embedding.shape[1])
        return self.scorer(torch.cat((involved, graph.action_features), dim=1)).squeeze(-1)

    def forward(self, graph):
        return self.score(graph, self.encode(graph))


class ActionMLP(nn.Module):
    """The existing control, using the same complete 16 action features."""

    def __init__(self, hidden=64, *, normalization=FeatureConfig()):
        super().__init__()
        self.normalization, self.trained = normalization, False
        self.scorer = nn.Sequential(nn.Linear(len(ACTION_FEATURES), hidden), nn.ELU(),
                                    nn.Linear(hidden, 1))

    def forward(self, graph):
        if graph.normalization != self.normalization:
            raise ValueError('graph/model normalization differs')
        return self.scorer(graph.action_features).squeeze(-1)


MODEL_VARIANTS = ('full_gnn', 'no_edge_gnn', 'action_mlp')


def make_ranker(variant='full_gnn', *, normalization=FeatureConfig()):
    if variant == 'action_mlp':
        return ActionMLP(normalization=normalization)
    if variant in ('full_gnn', 'no_edge_gnn'):
        return BottleneckGAT(use_edge_features=variant == 'full_gnn', normalization=normalization)
    raise ValueError(f'unknown ranking model: {variant}')


def route_identity(routes, bits):
    return tuple(tuple(map(int, r)) for r in routes), tuple(tuple(map(int, d)) for d in bits)


def directed_route_edges(state):
    edges = set()
    for r, (route, bits) in enumerate(zip(state.routes, state.bits)):
        jobs = [(int(n), int(d)) for n, d in zip(route, bits)]
        # Home/tail sentinels distinguish empty, single-job and ownership
        # changes too; interior edges are directed job/direction adjacencies.
        path = [(-1, 0), *jobs, (-2, 0)]
        edges.update((r, a, b) for a, b in zip(path, path[1:]))
    return edges


def route_distance(a, b):
    left, right = directed_route_edges(a), directed_route_edges(b)
    return 1. - len(left & right) / max(1, len(left | right))


def route_diversity(states):
    distances = [route_distance(a, b) for i, a in enumerate(states) for b in states[i + 1:]]
    return float(np.mean(distances)) if distances else 0.


def action_distribution(scores, temperature=1.):
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('non-finite action scores')
    if not len(values):
        return values, 0.
    weights = np.exp((values - values.max()) / temperature)
    probabilities = weights / weights.sum()
    entropy = (-float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.e-300))))
               / math.log(len(values))) if len(values) > 1 else 0.
    return probabilities, float(np.clip(entropy, 0., 1.))


def ranked_action_indices(scores, indices, budget, rng, *, uncertainty=None, temperature=1.):
    ordered = sorted(indices, key=lambda i: (-float(scores[i]), i))
    selected = ordered[:min(budget, len(ordered))]
    if uncertainty is None or len(ordered) < 2 or uncertainty < .5:
        return selected
    # Keep the highest score, then sample from the same bounded Top-K region.
    # Higher uncertainty raises the chance of replacing a greedy choice.
    available = ordered[1:max(budget, 4 * budget)]
    selected = ordered[:1]
    while available and len(selected) < budget:
        if rng.random() < uncertainty:
            probabilities, _ = action_distribution([scores[i] for i in available], temperature)
            chosen = rng.choices(available, weights=probabilities, k=1)[0]
        else:
            chosen = available[0]
        selected.append(chosen); available.remove(chosen)
    return selected


class FixedWorkspace:
    """Reuse static fixed-task numeric tables, never HGA's search routines."""

    def __init__(self, taskset, config=ScientificConfig(), *, normalization=FeatureConfig()):
        if not isinstance(taskset, FixedTaskset):
            raise TypeError('GNN-VNS-BR requires a FixedTaskset with immutable atom boundaries')
        self.taskset, self.config = taskset, config
        self.normalization = normalization
        self.numeric = _HGAWorkspace(taskset.parents, config, FORMAL_SCOPE_V2,
                                     patterns=taskset.patterns)
        w = self.numeric
        self.index = {identifier: i for i, identifier in enumerate(w.ids)}
        self.scale = max(1., sum(config.process_time(p.length) for p in taskset.parents) / 4.)
        self.lengths = np.linalg.norm(w.points[:, 1] - w.points[:, 0], axis=1)
        self.centers = w.points.mean(axis=1)
        self.evaluations = 0
        self._evaluated = {}
        self.initial_candidates = []
        self.initialization = {}
        # The graph order follows FixedTaskset.tasks, independently of the
        # numeric catalog of every possible merged job.
        self.atom_nodes = tuple(self.index[t.task_id] for t in taskset.tasks)
        self.atom_index = {n: i for i, n in enumerate(self.atom_nodes)}
        self.job_atoms = tuple(tuple(self.atom_index[int(n)] for n in
                              w.atomic_nodes[w.fixed_starts[j]:w.fixed_stops[j]])
                               for j in range(len(w.ids)))
        self.parent_edges = []
        parents = {}
        for i, task in enumerate(taskset.tasks):
            parents.setdefault(task.source_parent_id, []).append(i)
        for group in parents.values():
            group.sort(key=lambda i: taskset.tasks[i].u_start)
            self.parent_edges.extend(zip(group, group[1:]))
        self.spatial_edges = {}
        for n in self.atom_nodes:
            nearest = sorted((float(w.travel[n, m].min()), m) for m in self.atom_nodes if n != m)[:4]
            for _, m in nearest:
                for a, b in ((n, m), (m, n)):
                    self.spatial_edges[self.atom_index[a], self.atom_index[b]] = w.travel[a, b].reshape(4).copy()
        self.home_travel = np.empty((4, len(self.atom_nodes), 2))
        # Reuse the native MOVE template for initial TCP approach times. This
        # constructs motion data only; it never runs scheduling/coordination.
        for r in range(4):
            home = np.array(_home(r, config))
            for i, n in enumerate(self.atom_nodes):
                points = np.array([[home, home], w.points[n]])
                for bit in (0, 1):
                    ops = _numeric_templates(np.array([0, 1], np.int64), np.array([0, bit], np.int64),
                        points, np.array([config.process_time(0.), w.processing[n]]),
                        config.empty_speed, config.t_pre, config.t_post)
                    self.home_travel[r, i, bit] = ops[ops[:, 8] == 0, 0].sum()

    def costs(self, routes, bits):
        w = self.numeric
        costs = np.array([_oriented_route_cost(r, d, w.travel, w.processing)
                          for r, d in zip(routes, bits)])
        loads = np.array([w.processing[r].sum() for r in routes])
        return costs, loads

    def evaluate(self, routes, bits):
        key = route_identity(routes, bits)
        if key in self._evaluated:
            return self._evaluated[key]
        w = self.numeric
        try:
            self.validate(routes, bits)
        except (ValueError, IndexError):
            result = State(routes, bits, math.inf, np.zeros(4), np.zeros(4), np.zeros(4),
                           [np.empty((0, 9)) for _ in range(4)], ScheduleStatus.INFEASIBLE)
            self._evaluated[key] = result
            return result
        value, vectors, operations, completion = _priority_arrays(
            routes, w.travel, w.processing, w.points, w.parameters, bits)
        self.evaluations += 1
        costs, loads = self.costs(routes, vectors)
        status = ScheduleStatus.FEASIBLE if math.isfinite(value) else ScheduleStatus.DEADLOCK
        if (math.isnan(value) or not np.isfinite(costs).all() or not np.isfinite(completion).all()
                or any(not np.isfinite(ops).all() for ops in operations)):
            status = ScheduleStatus.NUMERIC_FAILURE
        result = State([r.copy() for r in routes], [d.copy() for d in vectors], float(value),
                       completion, costs, loads, operations, status)
        self._evaluated[key] = result
        return result

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
        result = _schedule_from_arrays(state.value, state.bits, state.operations, state.completion,
                                      self.numeric.ids, 'NATIVE_PRIORITY_FOUR_ROBOTS_EXPLICIT_DIRECTIONS')
        return replace(result, status=state.status, cmax=state.value if state.feasible else None)

    def validate(self, routes, bits):
        w = self.numeric
        if len(routes) != 4 or len(bits) != 4:
            raise ValueError('exactly four routes and direction vectors are required')
        coverage = np.zeros(len(w.atomic_nodes), np.int64)
        for r in range(4):
            if len(routes[r]) != len(bits[r]) or np.any((bits[r] != 0) & (bits[r] != 1)):
                raise ValueError('invalid direction genes')
            if np.any((routes[r] < 0) | (routes[r] >= len(w.ids))):
                raise ValueError('unknown merged job')
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

    def _construct_initial(self, source, rng):
        """Four small construction sources on the unchanged fixed atoms."""
        w = self.numeric
        routes = [np.empty(0, np.int64) for _ in range(4)]
        bits = [np.empty(0, np.int64) for _ in range(4)]
        if source == 'spatial':
            # Spatial greedy assignment; atom boundaries remain unchanged.
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
                boundary = int(np.searchsorted(prefix, prefix[-1] * .5)) + 1
                for side, owned in enumerate((nodes[:boundary], nodes[boundary:])):
                    r = 2 * rail + side
                    routes[r] = np.array(owned, np.int64)
                    bits[r] = np.zeros(len(owned), np.int64)
            routes, bits = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
            for r in range(4):
                pending = list(map(int, routes[r]))
                ordered, directions = [], []
                point = np.array(_home(r, self.config))
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
            if source == 'randomized_insertion':
                rng.shuffle(pending)
            elif source == 'restricted_first':
                pending.sort(key=lambda n: (int(w.eligible[n].sum()), -w.processing[n], n))
            else:
                pending.sort(key=lambda n: (-w.processing[n], n))
            for node in pending:
                if source == 'load_travel':
                    choices = []
                    for robot in np.flatnonzero(w.eligible[node]):
                        eligible = w.eligible.copy()
                        eligible[node] = False; eligible[node, robot] = True
                        r, position, bit = _insert_fixed_oriented(node, routes, bits, eligible,
                            w.travel, w.processing, w.fixed_starts, w.fixed_stops, w.fixed_lookup, False)
                        proposed, ds = list(routes), list(bits)
                        _put_oriented(proposed, ds, node, bit, r, position)
                        proposed, ds = _merge_fixed_arrays(proposed, ds, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
                        costs, loads = self.costs(proposed, ds)
                        choices.append((float(max(loads) + .5 * max(costs - loads)),
                                        float(max(costs)), int(r), proposed, ds))
                    _, _, _, routes, bits = min(choices, key=lambda item: item[:3])
                else:
                    r, position, bit = _insert_fixed_oriented(
                        node, routes, bits, w.eligible, w.travel, w.processing,
                        w.fixed_starts, w.fixed_stops, w.fixed_lookup, True)
                    _put_oriented(routes, bits, node, bit, r, position)
                    routes, bits = _merge_fixed_arrays(routes, bits, w.fixed_starts, w.fixed_stops, w.fixed_lookup)
        return _repair_oriented_first(routes, bits, w.points, w.travel, w.parameters, len(w.processing))

    def initialize(self, rng, settings, deadline, model=None):
        """At most six distinct starts, then the existing single-solution VNS."""
        pool, sources, seen = [], [], set()
        budget = min(6, settings.initial_attempts)
        def offer(source, routes, bits):
            key = route_identity(routes, bits)
            if key in seen:
                return
            seen.add(key)
            state = self.evaluate(routes, bits)
            if state.status in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                pool.append(state); sources.append(source)
        def quality(state):
            return (not state.feasible, state.value if state.feasible else float(max(state.costs)))
        attempted_sources = []
        for source in ('spatial', 'randomized_insertion', 'restricted_first', 'load_travel')[:budget]:
            if pool and time.perf_counter() >= deadline:
                break
            attempted_sources.append(source)
            offer(source, *self._construct_initial(source, rng))
        guided = isinstance(model, BottleneckGAT) and model.trained
        uncertainty = None
        # D decides whether to spend the remaining small construction budget
        # on structural shaking. No population is evolved or baseline read.
        for attempt in range(max(0, budget - 4)):
            if not pool or time.perf_counter() >= deadline:
                break
            if attempt and route_diversity(pool) >= .5:
                break
            base = min(pool, key=quality)
            actions = candidate_actions(self, base, settings, rng)
            actions = [a for a in actions if route_identity(a.routes, a.bits) not in seen]
            if not actions:
                break
            if guided and attempt == 0:
                with torch.inference_mode():
                    scores = model(local_graph(self, base, actions)).cpu().numpy()
                probabilities, uncertainty = action_distribution(scores, settings.temperature)
                top = sorted(range(len(actions)), key=lambda i: (-scores[i], i))[:settings.top_k]
                i = (rng.choices(top, weights=probabilities[top], k=1)[0]
                     if rng.random() < uncertainty else top[0])
                source = 'gnn_guided'
            else:
                i = rng.randrange(len(actions)); source = 'structural_perturbation'
            offer(source, actions[i].routes, actions[i].bits)
        if not pool:
            return None
        best = min(pool, key=quality)
        # Retain the best feasible state first. For similar quality, order the
        # other starts by distance to the retained routes, for later fallback.
        retained = [best]
        remaining = [s for s in pool if s is not best]
        while remaining:
            top = min(remaining, key=quality)
            peers = [s for s in remaining if s.feasible == top.feasible and
                     quality(s)[1] <= quality(top)[1] * 1.05 + 1.e-7]
            chosen = max(peers, key=lambda s: min(route_distance(s, old) for old in retained))
            retained.append(chosen); remaining = [s for s in remaining if s is not chosen]
        self.initial_candidates = retained
        self.initialization = dict(sources=sources, candidates=len(pool),
                                   attempted_sources=attempted_sources,
                                   diversity=route_diversity(pool), uncertainty=uncertainty,
                                   gnn_guidance=guided)
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
        source_empty = (costs[ru] - loads[ru]) - (state.costs[ru] - state.loads[ru])
        target_empty = (costs[rv] - loads[rv]) - (state.costs[rv] - state.loads[rv])
        feature[6:11] = np.array([
            loads[ru] - state.loads[ru], loads[rv] - state.loads[rv],
            source_empty, target_empty, max(costs) - max(state.costs)]) / self.normalization.time_ref
        feature[11] = w.processing[moved] / max(1., state.loads[ru])
        feature[12] = float(flip)
        feature[13 + {0: 0, 1: 1, -1: 2}[portion]] = 1.
        return Action(move, u, v, ru, rv, routes, bits, costs, loads, feature,
                      iu, iv, bool(flip), portion, moved)


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
                        offer(2, rv, iu, iv, flip=bool((ordinal + rv) % 2))
                        offer(2, rv, iu, iv, flip=not bool((ordinal + rv) % 2))
                if iu + 1 < len(route) and np.all(w.eligible[route[iu:iu + 2], rv]):
                    legal = [j for j in indices if j + 1 < len(target)
                             and np.all(w.eligible[target[j:j + 2], heavy])]
                    for iv in legal[:settings.alpha]:
                        offer(3, rv, iu, iv, flip=bool((ordinal + rv) % 2))
                        offer(3, rv, iu, iv, flip=not bool((ordinal + rv) % 2))
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
    """Complete atomic heterograph, shared by supervision and inference.

    Numeric merged-job IDs remain search metadata. Every graph task node is
    one immutable FixedTaskset.tasks entry. No scheduler is invoked here.
    """
    w = workspace.numeric
    norm = workspace.normalization
    robot_base = len(workspace.taskset.tasks)
    x = np.zeros((robot_base + 4, len(NODE_FEATURES)), np.float32)
    origin = np.array([norm.x_bounds[0], norm.y_bounds[0]])
    span = np.array([norm.x_bounds[1] - norm.x_bounds[0], norm.y_bounds[1] - norm.y_bounds[0]])
    def coordinates(points):
        return 2. * (np.asarray(points) - origin) / span - 1.
    owners = np.full(robot_base, -1, np.int64)
    for j, task in enumerate(workspace.taskset.tasks):
        x[j, 0] = 1.
        x[j, 2:6] = coordinates([task.start, task.end]).flatten()
        x[j, 6:9] = (task.length / norm.length_ref, task.u_start, task.u_end)
    for r in range(4):
        for i, node in enumerate(state.routes[r]):
            atoms = workspace.job_atoms[int(node)]
            owners[list(atoms)] = r
            x[list(atoms), 9:12] = (1., i / max(1, len(state.routes[r]) - 1), int(state.bits[r][i]))
    feedback = np.full(4, state.feasible, bool)
    approach = workspace.home_travel.copy()
    for r in range(4):
        x[robot_base + r, 1] = 1.
        x[robot_base + r, 2 + r] = 1.
        tcp = _home(r, workspace.config)
        if len(state.routes[r]):
            tail, bit = int(state.routes[r][-1]), int(state.bits[r][-1])
            tcp = w.points[tail, 1 - bit]
            approach[r] = w.travel[tail, list(workspace.atom_nodes), bit, :]
        x[robot_base + r, 6:8] = coordinates(tcp)
        ops = state.operations[r] if len(state.operations) == 4 else np.empty((0, 9))
        empty = float((ops[ops[:, 8] == 0, 6] - ops[ops[:, 8] == 0, 5]).sum())
        waiting = float((ops[ops[:, 8] == 4, 6] - ops[ops[:, 8] == 4, 5]).sum())
        # Partial timelines are not complete observations. The validity bit
        # is embedded separately, keeping both node types at twelve inputs.
        completion = float(state.completion[r]) if np.isfinite(state.completion[r]) else 0.
        if not feedback[r]:
            empty = completion = waiting = 0.
        x[robot_base + r, 8:12] = np.array([state.loads[r], empty, completion, waiting]) / norm.time_ref

    edges = {(i, i, 0): (0., 0., 0., 0.) for i in range(len(x))}
    for a, b in workspace.parent_edges:
        edges[a, b, 2] = (0., 0., 0., 0.)
        edges[b, a, 3] = (0., 0., 0., 0.)
    for (a, b), travel in workspace.spatial_edges.items():
        edges[a, b, 6] = tuple(travel / norm.time_ref)
    def endpoints(node, direction):
        atoms = workspace.job_atoms[int(node)]
        return (atoms[0], atoms[-1]) if direction == 0 else (atoms[-1], atoms[0])
    for r in range(4):
        for i in range(1, len(state.routes[r])):
            a, b = map(int, state.routes[r][i - 1:i + 1])
            da, db = map(int, state.bits[r][i - 1:i + 1])
            entry_a, exit_a = endpoints(a, da)
            entry_b, exit_b = endpoints(b, db)
            edges[exit_a, entry_b, 4] = (0., 0., float(w.travel[a, b, da, db]) / norm.time_ref, 0.)
            # Reversing a job transition uses the other job's exit and entry
            # atoms, which need not equal the forward endpoints.
            edges[exit_b, entry_a, 5] = (0., 0., float(w.travel[b, a, db, da]) / norm.time_ref, 0.)
    for j, n in enumerate(workspace.atom_nodes):
        for robot in range(4):
            attr = (float(w.eligible[n, robot]), float(owners[j] == robot),
                    *map(float, approach[robot, j] / norm.time_ref))
            edges[j, robot_base + robot, 1] = attr
            edges[robot_base + robot, j, 1] = attr
    ordered = sorted(edges)
    involved, membership = [], []
    for k, a in enumerate(actions):
        groups = (workspace.job_atoms[a.moved], workspace.job_atoms[a.v] if a.v >= 0 else (robot_base + a.rv,),
                  (robot_base + a.ru,), (robot_base + a.rv,))
        involved.append([group[0] for group in groups])
        membership.extend((n, 4 * k + j) for j, group in enumerate(groups) for n in group)
    return LocalGraph(
        torch.from_numpy(x),
        torch.tensor([(a, b) for a, b, _ in ordered], dtype=torch.long).t().contiguous(),
        torch.tensor([relation for _, _, relation in ordered], dtype=torch.long),
        torch.tensor([edges[key] for key in ordered], dtype=torch.float32),
        torch.tensor(involved, dtype=torch.long).reshape(-1, 4),
        torch.tensor(np.array([a.features for a in actions]), dtype=torch.float32).reshape(-1, len(ACTION_FEATURES)),
        torch.tensor(membership, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        tuple(t.task_id for t in workspace.taskset.tasks),
        tuple(t.source_parent_id for t in workspace.taskset.tasks), workspace.atom_nodes,
        torch.from_numpy(feedback), norm)


def ranking_pairs(values, rng, limit=256):
    """Only FEASIBLE values have numeric order; failure states have no Cmax.

    State inputs also carry structural INFEASIBLE and NUMERIC_FAILURE status.
    The numeric form remains available for callers with native scalar labels.
    """
    scalars = []
    for label in values:
        if isinstance(label, State):
            scalars.append(math.nan if label.status is ScheduleStatus.NUMERIC_FAILURE else
                           label.value if label.feasible else math.inf)
        else:
            scalars.append(label)
    values = scalars
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
    normalization = examples[0][1].normalization
    if any(graph.normalization != normalization for _, graph, _ in examples):
        raise ValueError('TRAIN graphs must share one fixed normalization')
    model = make_ranker(variant, normalization=normalization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); rng.shuffle(training)
        losses, gradient_norms = [], []
        for _, graph, pairs in training:
            scores = model(graph)
            loss = F.softplus(scores[pairs[:, 1]] - scores[pairs[:, 0]]).mean()
            if not torch.isfinite(scores).all() or not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite TRAIN scores/loss at epoch {epoch}')
            optimizer.zero_grad(); loss.backward()
            gradient_norms.append(float(nn.utils.clip_grad_norm_(
                model.parameters(), 5., error_if_nonfinite=True)))
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            held_losses, correct, total = [], 0, 0
            for _, graph, pairs in held:
                scores = model(graph)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(f'non-finite held-TRAIN scores at epoch {epoch}')
                delta = scores[pairs[:, 0]] - scores[pairs[:, 1]]
                held_losses.append(float(F.softplus(-delta).mean()))
                correct += int((delta > 0).sum()); total += len(delta)
        criterion = float(np.mean(held_losses if held_losses else losses))
        if criterion < best_loss:
            best_loss, best_epoch, best_state = criterion, epoch, copy.deepcopy(model.state_dict())
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                      held_train_loss=criterion, held_train_pair_accuracy=correct / max(1, total),
                      max_gradient_norm=max(gradient_norms))
        history.append(record); progress('BR_EPOCH', dict(model=variant, **record))
    model.load_state_dict(best_state); model.eval(); model.trained = True
    return model, dict(variant=variant, selected_epoch=best_epoch, history=history,
                       epochs=epochs, seed=seed, learning_rate=learning_rate,
                       parameters=sum(p.numel() for p in model.parameters()),
                       train_layouts=sorted(set(layouts) - holdout), held_train_layouts=sorted(holdout),
                       feature_names=NODE_FEATURES, robot_feature_names=ROBOT_FEATURES,
                       edge_relations=EDGE_RELATIONS, normalization=asdict(normalization),
                       action_feature_names=ACTION_FEATURES)


def run_gnn_vns_br(taskset, model, *, seed, settings=BRConfig(), config=ScientificConfig(), progress=None):
    """Independent initialization, bottleneck VNS and bounded guided shaking.

    Returns the best native-feasible state with its cached native timeline.
    The experiment entry point packages these arrays without rescheduling.
    """
    started = time.perf_counter()
    deadline = started + settings.time_limit
    rng = random.Random(seed)
    workspace = FixedWorkspace(taskset, config, normalization=getattr(model, 'normalization', FeatureConfig()))
    current = workspace.initialize(rng, settings, deadline, model)
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
    uncertainty = None
    use_uncertainty = isinstance(model, BottleneckGAT) and model.trained
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
        if use_uncertainty:
            _, uncertainty = action_distribution(scores, settings.temperature)
        inference_seconds += time.perf_counter() - tick
        encodings += 1
        improved = False
        evaluated = {}
        winner = None
        winner_move = None
        for move in range(1, 7):
            indices = [i for i, a in enumerate(actions) if a.move == move]
            selected = ranked_action_indices(scores, indices, settings.top_k, rng,
                uncertainty=uncertainty, temperature=settings.temperature)
            for i in selected:
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
            selected = ranked_action_indices(scores, large, settings.top_k, rng,
                uncertainty=uncertainty, temperature=settings.temperature)
            for i in selected:
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
                current = workspace.initialize(rng, settings, deadline, model)
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
        initialization=workspace.initialization, action_uncertainty=uncertainty,
        inference_and_graph_s=inference_seconds,
        generated_by_move=generated_by_move, evaluated_by_move=evaluated_by_move,
        feasible_by_move=feasible_by_move, accepted_by_move=accepted_by_move,
        initial_native_cmax_s=initial_value if math.isfinite(initial_value) else None,
        search_runtime_s=time.perf_counter() - started,
        termination_reason='TIME_LIMIT' if time.perf_counter() >= deadline else
                           'ROUND_LIMIT' if rounds >= settings.max_rounds else 'LOCAL_OPTIMUM',
        parameters=asdict(settings))
