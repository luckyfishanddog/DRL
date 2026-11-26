"""GNN-conditioned complete-plan sampling and NATIVE-elite cross entropy.

One probability model samples legal modes, then complete append-only directed
routes from empty state. Every elite is replayed in full, including its modes;
one Adam step maximizes its trajectory likelihood with entropy regularization.
No incumbent neighborhood, legacy action head, native scorer or local optimizer
is used. The interleaving of four routes is an auxiliary sampled trajectory;
we fit that trajectory likelihood, not a sum over all equivalent interleavings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mrta_baselines.hga import _HGAWorkspace
from mrta_ranker.model import build_static_graph
from mrta_reference.geometry import blocks_for_pattern
from mrta_reference.model import ScientificConfig, OperationKind
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.scheduler import native_schedule
from mrta_search.gnn_constructor import VNextGNN, build_block_robot_graph
from mrta_search.saeax import _heads_legal, _route_statistics, validate_individual, complete_identity


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 8
    max_generations: int = 3
    elite_fraction: float = .25
    max_new_coordination: int = 24
    entropy_beta: float = .01
    exploration_mix: float = .05
    learning_rate: float = .001
    gradient_clip: float = 5.


@dataclass
class Trace:
    selected: tuple[int, ...]
    actions: tuple[tuple[int, int, int], ...]  # workspace child, robot, direction


@dataclass
class Plan:
    solution: object
    directions: tuple
    trace: Trace
    costs: np.ndarray
    loads: np.ndarray
    key: tuple
    schedule: object = None
    origin: str = ''


def plan_from_arrays(w, selected, routes, directions, actions=None):
    rr = [np.array(row, np.int64) for row in routes]
    dd = [np.array(row, np.int64) for row in directions]
    selected = np.array(selected, np.int64)
    validate_individual(w, rr, dd, selected)
    if not _heads_legal(rr, dd, w.points, w.parameters):
        raise ValueError('illegal complete-plan first tasks')
    solution = w.solution(rr, selected)
    expected = tuple(tuple(w.ids[int(n)] for n in row) for row in rr)
    if (tuple(r.block_ids for r in solution.routes) != expected or
            solution.patterns != tuple(w.patterns[int(m)] for m in selected)):
        raise ValueError('canonicalization changed the sampled modes/routes')
    if actions is None:
        actions = [(int(rr[r][k]), r, int(dd[r][k]))
            for k in range(max(map(len, rr), default=0)) for r in range(4) if k < len(rr[r])]
    costs, loads = _route_statistics(rr, dd, w.travel, w.processing)
    return Plan(solution, tuple(tuple(map(int, row)) for row in dd),
        Trace(tuple(map(int, selected)), tuple(actions)), costs, loads,
        complete_identity(rr, dd, selected))


def plan_from_solution(w, solution, directions):
    return plan_from_arrays(w, [w.patterns.index(p) for p in solution.patterns], w.encode(solution), directions)


def solution_record(plan):
    if plan is None: return None
    return dict(patterns=[dict(parent_id=p.parent_id, kind=p.kind.value, t=p.t,
        point_id=p.point_id, rail=p.rail.value if p.rail else None, mandatory=p.mandatory)
        for p in plan.solution.patterns], routes=[list(r.block_ids) for r in plan.solution.routes],
        directions=plan.directions)


@dataclass
class Prefix:
    routes: list = field(default_factory=lambda: [[] for _ in range(4)])
    bits: list = field(default_factory=lambda: [[] for _ in range(4)])
    costs: np.ndarray = field(default_factory=lambda: np.zeros(4))
    loads: np.ndarray = field(default_factory=lambda: np.zeros(4))
    used: set = field(default_factory=set)

    def append(self, w, action):
        n, r, d = action
        if self.routes[r]: self.costs[r] += w.travel[self.routes[r][-1], n, self.bits[r][-1], d]
        self.loads[r] += w.processing[n]; self.costs[r] += w.processing[n]
        self.routes[r].append(n); self.bits[r].append(d); self.used.add(n)


def legal_actions(w, active, prefix):
    """Exact immediate eligibility, canonical adjacency and original head mask.

    This mask is not a completion oracle. A legal prefix can reach a dead end;
    such a sample consumes its one sampling slot, without repair or retries.
    """
    pending = [n for n in active if n not in prefix.used]
    choices = []
    for n in pending:
        for r in range(4):
            if not w.eligible[n, r]: continue
            if prefix.routes[r] and w.parent_index[prefix.routes[r][-1]] == w.parent_index[n]: continue
            for d in (0, 1):
                if not prefix.routes[r]:
                    rr = [np.array(row[:1], np.int64) for row in prefix.routes]
                    dd = [np.array(row[:1], np.int64) for row in prefix.bits]
                    rr[r], dd[r] = np.array([n], np.int64), np.array([d], np.int64)
                    if not _heads_legal(rr, dd, w.points, w.parameters): continue
                choices.append((int(n), r, d))
    return choices


class TaskSpace:
    def __init__(self, world, selected):
        w = self.w = world.w
        self.selected = tuple(selected)
        self.active = tuple(int(n) for m in selected for n in w.pattern_nodes[m] if n >= 0)
        self.parents = len(w.parents); self.robot_offset = self.parents+len(self.active)
        self.child_index = {n: self.parents+i for i, n in enumerate(self.active)}
        blocks = tuple(b for parent, m in zip(w.parents, selected)
            for b in blocks_for_pattern(parent, w.patterns[m], w.config))
        # Only activated children exist in this graph. Alternatives stay in the
        # legal geometry catalog, never as inactive padding in the task graph.
        child = build_block_robot_graph(blocks, tuple(w.patterns[m] for m in selected),
            w.config, include_parent_edges=True)
        self.features = torch.cat((world.parent_features, child.node_features), dim=0)
        self.features[self.parents:self.robot_offset, 14:18] = torch.tensor(w.eligible[list(self.active)], dtype=torch.float32)
        edges, relations, known = [], [], set()
        def edge(a, b, t):
            key = a, b, t
            if key not in known:
                edges.append((a, b)); relations.append(t); known.add(key)
        for (a, b), t in zip(world.parent_graph.edge_index.T.tolist(), world.parent_graph.edge_type.tolist()):
            edge(a if a < self.parents else a+len(self.active),
                 b if b < self.parents else b+len(self.active), t)
        for (a, b), t in zip(child.edge_index.T.tolist(), child.edge_type.tolist()):
            if t == 3:
                node, robot = (a, b-len(self.active)) if a < len(self.active) else (b, a-len(self.active))
                if not w.eligible[self.active[node], robot]: continue
            edge(a+self.parents, b+self.parents, t)
        for n in self.active:
            p, i = int(w.parent_index[n]), self.child_index[n]
            edge(p, i, 4); edge(i, p, 4)
        self.edges, self.relations = edges, relations
        self.parent_counts = np.bincount(w.parent_index[list(self.active)], minlength=self.parents)

    def graph(self, prefix):
        w, p = self.w, self.parents
        rows = torch.zeros((len(self.features), 32), dtype=torch.float32)
        rows[:, :20] = self.features
        rows[:p, 31] = 1.
        for n in self.active:
            rows[self.child_index[n], 29] = float(n not in prefix.used)
            rows[int(w.parent_index[n]), 29] += float(n not in prefix.used)/self.parent_counts[w.parent_index[n]]
        edges, types = list(self.edges), list(self.relations)
        peak = int(np.argmax(prefix.costs))
        for r in range(4):
            robot = self.robot_offset+r
            rows[robot, 20+r] = 1.
            rows[robot, 26:29] = torch.tensor([prefix.loads[r], prefix.costs[r]-prefix.loads[r], prefix.costs[r]])/2500.
            rows[robot, 30] = float(r == peak)
            for k, (n, d) in enumerate(zip(prefix.routes[r], prefix.bits[r])):
                i, parent = self.child_index[n], int(w.parent_index[n])
                rows[i, 20+r] = 1.; rows[parent, 20+r] += 1./self.parent_counts[parent]
                rows[i, 24:26] = torch.tensor([k/max(1, len(prefix.routes[r])-1), d])
                rows[i, 26:29] = rows[robot, 26:29]; rows[i, 30] = float(r == peak)
                edges.extend(((i, robot), (robot, i))); types.extend((10, 10))
                if k:
                    before = self.child_index[prefix.routes[r][k-1]]
                    edges.extend(((before, i), (i, before))); types.extend((6, 7))
        return SimpleNamespace(node_features=rows, edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
            edge_type=torch.tensor(types, dtype=torch.long))


class World:
    def __init__(self, w):
        self.w = w
        self.parent_graph = build_static_graph(w.parents, w.config)
        self.parent_features = []
        for p, parent in enumerate(w.parents):
            delta = np.array(parent.end)-parent.start; length = parent.length
            nodes = [int(n) for m in w.options[p] for n in w.pattern_nodes[m] if n >= 0]
            eligible = np.any(w.eligible[nodes], axis=0)
            self.parent_features.append([1., 0., parent.start[0]/20., parent.start[1]/12.,
                parent.end[0]/20., parent.end[1]/12., (parent.start[0]+parent.end[0])/40.,
                (parent.start[1]+parent.end[1])/24., delta[0]/20., delta[1]/12., length/24.,
                delta[0]/length, delta[1]/length, w.config.process_time(length)/2500.,
                *eligible.astype(float), float(all(w.patterns[m].mandatory for m in w.options[p])), 0.])
        self.parent_features = torch.tensor(self.parent_features, dtype=torch.float32).reshape(-1, 20)
        robot_rows = torch.zeros((4, 20)); robot_rows[:, 1] = 1.
        for r in range(4): robot_rows[r, 14+r] = 1.; robot_rows[r, 19] = r/3.
        features = torch.nn.functional.pad(torch.cat((self.parent_features, robot_rows)), (0, 12))
        features[:len(w.parents), 31] = 1.
        self.mode_graph = SimpleNamespace(node_features=features, edge_index=self.parent_graph.edge_index,
            edge_type=self.parent_graph.edge_type)
        parents, rows = [], []
        for p, options in enumerate(w.options):
            for m in options:
                pattern = w.patterns[m]; nodes = [int(n) for n in w.pattern_nodes[m] if n >= 0]
                possible = np.any(w.eligible[nodes], axis=0); work = np.zeros(4)
                for n in nodes: work += w.eligible[n]*w.processing[n]/sum(w.eligible[n])/2500.
                same_kind = sum(w.patterns[k].kind == pattern.kind for k in options)
                rows.append([len(nodes), sum(w.processing[nodes])/50., pattern.t if pattern.t is not None else .5,
                    float(pattern.mandatory), *(float(pattern.kind.value == k) for k in ('WHOLE', 'Y_SPLIT', 'X_SPLIT')),
                    *possible.astype(float), *work, math.log(same_kind)])
                parents.append(p)
        self.mode_features = torch.tensor(rows, dtype=torch.float32).reshape(-1, 16)
        self.mode_parents = torch.tensor(parents, dtype=torch.long)
        self.spaces = {}

    def space(self, selected):
        key = tuple(selected)
        if key not in self.spaces: self.spaces[key] = TaskSpace(self, key)
        return self.spaces[key]


class ModeHead(nn.Module):
    """Mode-dependent geometry interaction; a plain linear concatenation
    would cancel the identical parent embedding inside each parent's softmax.
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 64)
        self.score = nn.Linear(2*64+16, 1)

    def forward(self, rows):
        numeric = rows[:, -16:]
        gate_input = numeric.clone(); gate_input[:, 1] = gate_input[:, 1]/10.
        parent = rows[:, :64]*(1+self.gate(gate_input).tanh())
        return self.score(torch.cat((parent, rows[:, 64:128], numeric), dim=1))


class CEMPolicy(nn.Module):
    def __init__(self, encoder=None, exploration=.05):
        super().__init__()
        self.encoder = encoder or VNextGNN().enable_conflict_graph()
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(name.startswith(('layer1.', 'layer2.', 'geometry_skip.')))
        self.mode_head = ModeHead()
        self.route_head = nn.Linear(4*64+32, 1)
        self.exploration = exploration
        # Trainable cost priors initialize the new probability heads. They are
        # not fixed candidate ranking/selection or a legacy action head.
        with torch.no_grad():
            self.mode_head.score.weight.normal_(0., .005); self.mode_head.score.bias.zero_()
            self.mode_head.score.weight[:, -16:] = 0.
            self.mode_head.score.weight[0, -15] = -1.  # extra split setup processing
            self.mode_head.score.weight[0, -1] = -1.   # do not favor kinds with more cuts
            self.route_head.weight.normal_(0., .005); self.route_head.bias.zero_()
            self.route_head.weight[:, -32:] = 0.
            for i, value in ((0, .25), (1, -1.), (2, -1.), (3, -.5)):
                self.route_head.weight[0, -32+i] = value

    def probabilities(self, logits):
        return (1-self.exploration)*torch.softmax(logits, dim=0)+self.exploration/len(logits)

    def modes(self, world):
        z = self.encoder.encode_static(world.mode_graph)
        p = len(world.w.parents)
        robot = z[p:p+4].mean(0).expand(len(world.mode_features), -1)
        logits = self.mode_head(torch.cat((z[world.mode_parents], robot, world.mode_features), dim=1)).squeeze(1)
        return [self.probabilities(logits[list(options)]) for options in world.w.options]

    def actions(self, space, prefix, choices):
        w = space.w
        z = self.encoder.encode_static(space.graph(prefix))
        a = np.array(choices, np.int64); ns, rs, ds = a.T
        size = len(choices); last = np.array([prefix.routes[r][-1] if prefix.routes[r] else -1 for r in rs])
        ld = np.array([prefix.bits[r][-1] if prefix.bits[r] else 0 for r in rs])
        move = np.array([w.travel[n0, n, d0, d] if n0 >= 0 else 0. for n0, n, d0, d in zip(last, ns, ld, ds)])
        increment = w.processing[ns]+move
        projected = np.maximum(max(prefix.costs), prefix.costs[rs]+increment)-max(prefix.costs)
        endpoint = np.array([w.points[n, 1-d] if n >= 0 else (0., 0.) for n, d in zip(last, ld)])
        pending = len(space.active)-len(prefix.used)
        constant = lambda value: np.full(size, value)
        features = np.column_stack((w.processing[ns]/200., move/20.,
            (prefix.costs[rs]+increment-min(prefix.costs))/200., projected/200., prefix.costs[rs]/200.,
            np.broadcast_to(prefix.costs/200., (size, 4)), np.broadcast_to(prefix.loads/200., (size, 4)),
            constant(sum(w.processing[n] for n in space.active if n not in prefix.used)/4000.),
            constant(pending/len(space.active)), ds, w.points[ns, ds]/[20., 12.],
            w.points[ns, 1-ds]/[20., 12.], endpoint/[20., 12.], w.processing[ns]/200.,
            np.array([len(prefix.routes[r])/len(space.active) for r in rs]), w.eligible[ns],
            w.eligible[ns].sum(1)/4., constant(min(prefix.costs)/200.), rs == int(np.argmax(prefix.costs)), last >= 0))
        numeric = torch.tensor(features, dtype=torch.float32).reshape(-1, 32)
        child = torch.tensor([space.child_index[int(n)] for n in ns])
        robot = torch.tensor(rs+space.robot_offset)
        previous = torch.tensor([space.child_index[int(n)] if n >= 0 else int(r)+space.robot_offset for n, r in zip(last, rs)])
        logits = self.route_head(torch.cat((z[child], z[robot], z[w.parent_index[ns]], z[previous], numeric), dim=1)).squeeze(1)
        return self.probabilities(logits)


def load_policy(checkpoint, seed=20261110, exploration=.05):
    torch.manual_seed(seed)
    encoder = VNextGNN()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    keys = {k for k in encoder.state_dict() if k.startswith(('layer1.', 'layer2.', 'geometry_skip.'))}
    state = {k: v for k, v in saved.items() if k in keys}
    if set(state) != keys: raise ValueError('incomplete existing geometry encoder')
    encoder.load_state_dict(state, strict=False)
    return CEMPolicy(encoder.enable_conflict_graph(), exploration)


def entropy(probabilities):
    return -(probabilities*probabilities.clamp_min(1.e-30).log()).sum()


def draw(world, policy, generator, *, forced_modes=None):
    w = world.w
    with torch.no_grad():
        modes = policy.modes(world)
        selected = tuple(int(options[int(torch.multinomial(prob, 1, generator=generator))])
            for options, prob in zip(w.options, modes)) if forced_modes is None else tuple(forced_modes)
        space = world.space(selected); prefix = Prefix(); actions = []
        entropies = [float(entropy(p))/math.log(max(2, len(p))) for p in modes]
        while len(prefix.used) < len(space.active):
            choices = legal_actions(w, space.active, prefix)
            if not choices:
                pending = [n for n in space.active if n not in prefix.used]
                adjacency = sum(bool(prefix.routes[r]) and w.parent_index[prefix.routes[r][-1]] == w.parent_index[n]
                    for n in pending for r in range(4) if w.eligible[n, r])
                head_pairs = sum(not bool(prefix.routes[r]) for n in pending for r in range(4) if w.eligible[n, r])
                return None, dict(reason='NO_LEGAL_APPEND', assigned=len(prefix.used), active_children=len(space.active),
                    graph_nodes=len(space.features), normalized_entropy=float(np.mean(entropies)),
                    mode_entropy=float(np.mean(entropies[:len(modes)])),
                    action_entropy=float(np.mean(entropies[len(modes):])) if len(entropies)>len(modes) else None,
                    remaining_single_robot_children=int(sum(w.eligible[n].sum() == 1 for n in pending)),
                    adjacency_blocked_pairs=int(adjacency), head_blocked_pairs=int(head_pairs))
            prob = policy.actions(space, prefix, choices)
            if not torch.isfinite(prob).all(): raise ArithmeticError('non-finite CEM probabilities')
            action = choices[int(torch.multinomial(prob, 1, generator=generator))]
            entropies.append(float(entropy(prob))/math.log(max(2, len(prob))))
            prefix.append(w, action); actions.append(action)
        plan = plan_from_arrays(w, selected, prefix.routes, prefix.bits, actions)
        return plan, dict(reason='COMPLETE', active_children=len(space.active),
            graph_nodes=len(space.features), normalized_entropy=float(np.mean(entropies)),
            mode_entropy=float(np.mean(entropies[:len(modes)])),
            action_entropy=float(np.mean(entropies[len(modes):])))


def trajectory_logp(world, policy, trace):
    probabilities = policy.modes(world)
    value = sum(p[list(options).index(m)].log() for options, p, m in zip(world.w.options, probabilities, trace.selected))
    prefix, space = Prefix(), world.space(trace.selected)
    for action in trace.actions:
        choices = legal_actions(world.w, space.active, prefix)
        if action not in choices: raise ValueError('elite action not legal in its exact prefix')
        value = value+policy.actions(space, prefix, choices)[choices.index(action)].log()
        prefix.append(world.w, action)
    if len(prefix.used) != len(space.active): raise ValueError('incomplete elite trace')
    return value


def cem_update(world, policy, optimizer, elites, settings):
    """Exact accumulated full-trajectory gradient, one parameter update.

    Prefix graphs are released after each factor's backward pass. Parameters
    stay unchanged until ALL mode/action factors of ALL elites are processed.
    This is maximum likelihood, not a policy-gradient/reward update.
    """
    if not elites or any(p.schedule is None or not p.schedule.feasible for p in elites):
        raise ValueError('CEM elites must have cached feasible NATIVE schedules')
    trained = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in trained}
    optimizer.zero_grad(set_to_none=True)
    nll, total_entropy, factors = 0., 0., 0
    for elite in elites:
        probabilities = policy.modes(world)
        mode_nll = sum(-p[list(options).index(m)].log() for options, p, m in
            zip(world.w.options, probabilities, elite.trace.selected))
        mode_entropy = sum(entropy(p) for p in probabilities)
        ((mode_nll-settings.entropy_beta*mode_entropy)/len(elites)).backward()
        nll += float(mode_nll.detach()); total_entropy += float(mode_entropy.detach()); factors += len(probabilities)
        prefix, space = Prefix(), world.space(elite.trace.selected)
        for action in elite.trace.actions:
            choices = legal_actions(world.w, space.active, prefix)
            if action not in choices: raise ValueError('elite prefix/mask mismatch')
            prob = policy.actions(space, prefix, choices)
            cost, ent = -prob[choices.index(action)].log(), entropy(prob)
            ((cost-settings.entropy_beta*ent)/len(elites)).backward()
            nll += float(cost.detach()); total_entropy += float(ent.detach()); factors += 1
            prefix.append(world.w, action)
        if len(prefix.used) != len(space.active): raise ValueError('partial elite trajectory')
    gradients = {prefix: math.sqrt(sum(float(p.grad.square().sum()) for n, p in trained
        if n.startswith(prefix) and p.grad is not None)) for prefix in ('encoder.', 'mode_head.', 'route_head.')}
    torch.nn.utils.clip_grad_norm_([p for n, p in trained], settings.gradient_clip)
    optimizer.step()
    differences = {prefix: math.sqrt(sum(float((p.detach()-before[n]).square().sum()) for n, p in trained
        if n.startswith(prefix))) for prefix in gradients}
    return dict(elites=len(elites), factors=factors, negative_log_likelihood=nll/len(elites),
        entropy_per_factor=total_entropy/factors, gradient_norm=gradients, parameter_delta_l2=differences)


@dataclass
class CEMResult:
    best: Plan | None
    initial: Plan | None
    runtime: float
    generations: int
    termination_reason: str
    accounting: dict
    progress: list


def evaluate(plan, w):
    result = native_schedule(plan.solution, w.config, scope=FORMAL_SCOPE_V2,
        orientations=dict(enumerate(plan.directions)))
    if result.feasible and result.directions != plan.directions:
        raise ValueError('NATIVE changed the sampled directions')
    plan.schedule = result
    return result


def run_cem(parents, checkpoint, initial_builder, *, seed=20261110, settings=None,
            progress_callback=None, accounting=None):
    settings = settings or CEMConfig()
    started = time.perf_counter()
    a = accounting if accounting is not None else {}
    a.update(initial_coordination_calls=0, new_coordination_calls=0, coordination_calls=0, sample_attempts=0,
        complete_samples=0, construction_failures=0, duplicates=0, accepted_elites=0, updates=0)
    raw_initial, initial_meta = initial_builder(parents, ScientificConfig(), seed)
    a.update(initial_coordination_calls=initial_meta['coordination_calls'], coordination_calls=initial_meta['coordination_calls'], initial=initial_meta)
    w = _HGAWorkspace(parents, ScientificConfig(), FORMAL_SCOPE_V2)
    world, policy = World(w), load_policy(checkpoint, seed, settings.exploration_mix)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=settings.learning_rate)
    rng = torch.Generator().manual_seed(seed)
    initial, cache, best = None, {}, None
    if raw_initial is not None:
        initial = plan_from_solution(w, raw_initial.solution, raw_initial.directions)
        initial.schedule, initial.origin = raw_initial.schedule, 'initial'
        cache[initial.key] = initial
        if initial.schedule.feasible: best = initial
    a['initialization_s'] = time.perf_counter()-started
    previous_elites = [initial] if best is not None else []
    if progress_callback:
        progress_callback(dict(phase='INITIAL', status=initial.schedule.status.value if initial else 'CONSTRUCTION_FAILED',
            cmax=initial.schedule.cmax if initial is not None and initial.schedule.feasible else None,
            coordination_calls=a['coordination_calls'], elapsed_s=time.perf_counter()-started))
    progress, generations = [], 0
    reason = 'GENERATION_LIMIT'
    for generation in range(1, settings.max_generations+1):
        population, observations = {}, []
        for ordinal in range(settings.population_size):
            a['sample_attempts'] += 1
            plan, info = draw(world, policy, rng)
            info.update(sample=ordinal+1)
            if plan is None:
                a['construction_failures'] += 1; info['status'] = 'CONSTRUCTION_FAILED'
                observations.append(info); continue
            a['complete_samples'] += 1; plan.origin = f'g{generation}:s{ordinal+1}'
            reused = plan.key in cache
            if reused:
                a['duplicates'] += 1; plan.schedule = cache[plan.key].schedule
            else:
                if a['new_coordination_calls'] >= settings.max_new_coordination:
                    reason = 'NEW_NATIVE_BUDGET'; break
                a['new_coordination_calls'] += 1; a['coordination_calls'] += 1
                evaluate(plan, w); cache[plan.key] = plan
            population.setdefault(plan.key, plan)
            feasible = plan.schedule.feasible
            if feasible and (best is None or plan.schedule.cmax < best.schedule.cmax-1.e-6): best = plan
            info.update(status=plan.schedule.status.value, cmax=plan.schedule.cmax if feasible else None,
                reused=reused, route_period=float(max(plan.costs)), pmax=float(max(plan.loads)),
                move=float(sum(plan.costs-plan.loads)), reverse_directions=sum(d for row in plan.directions for d in row),
                modes={kind: sum(w.patterns[m].kind.value == kind for m in plan.trace.selected)
                    for kind in ('WHOLE', 'Y_SPLIT', 'X_SPLIT')})
            observations.append(info)
        feasible = sorted((p for p in population.values() if p.schedule.feasible), key=lambda p: (p.schedule.cmax, p.key))
        elite_count = max(1, math.ceil(settings.population_size*settings.elite_fraction))
        elites = feasible[:elite_count] if feasible else previous_elites
        update = cem_update(world, policy, optimizer, elites, settings) if elites else None
        if update is not None:
            a['updates'] += 1; a['accepted_elites'] += len(elites)
            previous_elites = elites
        generations = generation
        row = dict(generation=generation, samples=observations, feasible_unique=len(feasible),
            elite_origins=[p.origin for p in elites], elite_cmax=[p.schedule.cmax for p in elites],
            retained_previous_elites=not bool(feasible), update=update,
            best_cmax=best.schedule.cmax if best else None, coordination_calls=a['coordination_calls'], elapsed_s=time.perf_counter()-started)
        progress.append(row)
        if progress_callback: progress_callback(dict(phase='GENERATION', **{k:v for k,v in row.items() if k != 'samples'}))
        if reason == 'NEW_NATIVE_BUDGET': break
    if best is None: reason = 'NO_FEASIBLE_COMPLETE_PLAN'
    a.update(unique_evaluated=len(cache), search_complete=True,
        initial_cmax=initial.schedule.cmax if initial is not None and initial.schedule.feasible else None,
        best_origin=best.origin if best else None, returned_initial=best is initial,
        final_route_period=float(max(best.costs)) if best else None,
        final_wait=sum(op.duration for op in best.schedule.operations if op.kind is OperationKind.WAIT) if best else None)
    return CEMResult(best, initial, time.perf_counter()-started, generations, reason, a, progress)
