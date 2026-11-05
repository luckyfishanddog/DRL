"""Block-level GNN policy for fixed-split task assignment.

This module deliberately does not depend on the historical candidate rankers.  The
GAT encoder sees the actual welding blocks produced by the frozen first-round split
rule, and the policy head scores eligible robots while an assignment is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
import time
from typing import Mapping, Sequence

import torch
from torch import nn

from mrta_ranker.model import SparseGATLayer
from mrta_reference.geometry import (
    blocks_for_pattern,
    build_legal_pattern_catalog,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    ParentWeld,
    Rail,
    Route,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize, block_map


NODE_DIM = 20
DYNAMIC_DIM = 13


@dataclass(frozen=True)
class BlockRobotGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    blocks: tuple[WeldingBlock, ...]
    block_index: Mapping[str, int]
    robot_offset: int

    @cached_property
    def parent_block_indices(self):
        grouped = {}
        for index, block in enumerate(self.blocks):
            grouped.setdefault(block.parent_id, []).append(index)
        return grouped

    @cached_property
    def incoming_edges(self):
        destination = self.edge_index[1]
        return tuple(torch.where(destination == node)[0] for node in range(len(self.node_features)))


@dataclass(frozen=True)
class ConstructionResult:
    solution: CanonicalSolution
    inference_seconds: float
    legal_action_fraction: float
    assignments: tuple[tuple[str, int], ...]


def fixed_split_patterns(
    parents: Sequence[ParentWeld], config: ScientificConfig
) -> tuple[SplitPattern, ...]:
    """Choose WHOLE whenever legal, otherwise the first catalogued mandatory Y split."""
    ordered = tuple(sorted(parents, key=lambda item: item.parent_id))
    catalog = build_legal_pattern_catalog(ordered, config, FORMAL_SCOPE_V2)
    result: list[SplitPattern] = []
    for parent in ordered:
        options = catalog[parent.parent_id]
        if whole_eligible_rails(parent.start, parent.end, config):
            result.append(next(item for item in options if item.kind is SplitKind.WHOLE))
            continue
        mandatory_y = tuple(
            item
            for item in options
            if item.kind is SplitKind.Y_SPLIT and item.mandatory
        )
        if not mandatory_y:
            raise ValueError(f"{parent.parent_id}: fixed split rule has no mandatory Y option")
        result.append(mandatory_y[0])
    return tuple(result)


def fixed_blocks(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
) -> tuple[WeldingBlock, ...]:
    by_parent = {parent.parent_id: parent for parent in parents}
    blocks = tuple(
        block
        for pattern in patterns
        for block in blocks_for_pattern(by_parent[pattern.parent_id], pattern, config)
    )
    return tuple(
        sorted(
            blocks,
            key=lambda block: (-config.process_time(block.length), block.block_id),
        )
    )


def _eligible_robots(block: WeldingBlock, config: ScientificConfig) -> tuple[int, ...]:
    robots = tuple(robot for robot in range(4) if robot_is_eligible(block, robot, config))
    if not robots:
        raise ValueError(f"{block.block_id}: no eligible robot")
    return robots


def build_block_robot_graph(
    blocks: Sequence[WeldingBlock],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
    *,
    include_weld_edges: bool = True,
    include_parent_edges: bool = False,
) -> BlockRobotGraph:
    """Build block-nearness, same-rail-risk and block-to-eligible-robot edges."""
    ordered = tuple(blocks)
    forced = {pattern.parent_id: pattern.mandatory for pattern in patterns}
    rows: list[list[float]] = []
    midpoints: list[tuple[float, float]] = []
    eligibilities: list[tuple[int, ...]] = []
    for block in ordered:
        mx = (block.start[0] + block.end[0]) / 2.0
        my = (block.start[1] + block.end[1]) / 2.0
        dx = block.end[0] - block.start[0]
        dy = block.end[1] - block.start[1]
        length = block.length
        eligible = _eligible_robots(block, config)
        midpoints.append((mx, my))
        eligibilities.append(eligible)
        rows.append(
            [
                1.0,
                0.0,
                block.start[0] / 20.0,
                block.start[1] / 12.0,
                block.end[0] / 20.0,
                block.end[1] / 12.0,
                mx / 20.0,
                my / 12.0,
                dx / 20.0,
                dy / 12.0,
                length / 24.0,
                dx / length,
                dy / length,
                config.process_time(length) / 2500.0,
                *(float(robot in eligible) for robot in range(4)),
                float(forced[block.parent_id]),
                0.0,
            ]
        )
    for robot in range(4):
        one_hot = [float(index == robot) for index in range(4)]
        rows.append([0.0, 1.0, *([0.0] * 12), *one_hot, 0.0, robot / 3.0])

    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []

    def add(source: int, destination: int, relation: int) -> None:
        edges.append((source, destination))
        edge_types.append(relation)

    node_count = len(ordered) + 4
    for node in range(node_count):
        add(node, node, 0)
    if include_weld_edges:
        for index, midpoint in enumerate(midpoints):
            nearest = sorted(
                (math.dist(midpoint, other), other_index)
                for other_index, other in enumerate(midpoints)
                if other_index != index
            )[:4]
            for _, other_index in nearest:
                add(index, other_index, 1)
        for left in range(len(ordered)):
            left_x = sorted((ordered[left].start[0], ordered[left].end[0]))
            left_rails = {0 if robot < 2 else 1 for robot in eligibilities[left]}
            related: list[tuple[float, int]] = []
            for right in range(len(ordered)):
                if left == right:
                    continue
                right_rails = {0 if robot < 2 else 1 for robot in eligibilities[right]}
                if not left_rails.intersection(right_rails):
                    continue
                right_x = sorted((ordered[right].start[0], ordered[right].end[0]))
                gap = max(0.0, max(left_x[0], right_x[0]) - min(left_x[1], right_x[1]))
                if gap <= config.interference_dx + config.numeric_epsilon:
                    related.append((math.dist(midpoints[left], midpoints[right]), right))
            for _, right in sorted(related)[:4]:
                add(left, right, 2)
    robot_offset = len(ordered)
    if include_parent_edges:
        known = set(zip(edges, edge_types))
        for i, left in enumerate(ordered):
            for j, right in enumerate(ordered):
                if i != j and left.parent_id == right.parent_id and ((i,j),2) not in known:
                    add(i,j,2)
    for block_index, eligible in enumerate(eligibilities):
        for robot in eligible:
            add(block_index, robot_offset + robot, 3)
            add(robot_offset + robot, block_index, 3)
    return BlockRobotGraph(
        node_features=torch.tensor(rows, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        blocks=ordered,
        block_index={block.block_id: index for index, block in enumerate(ordered)},
        robot_offset=robot_offset,
    )


class BlockAssignmentGNN(nn.Module):
    """Two-layer, hidden-64 GAT with a lightweight dynamic assignment head."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.layer1 = SparseGATLayer(NODE_DIM, hidden_dim, heads)
        self.layer2 = SparseGATLayer(hidden_dim, hidden_dim, heads)
        self.policy = nn.Sequential(
            nn.Linear(2 * hidden_dim + DYNAMIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_static(self, graph: BlockRobotGraph) -> torch.Tensor:
        hidden = torch.nn.functional.elu(
            self.layer1(graph.node_features, graph.edge_index, graph.edge_type,
                        incoming_edges=graph.incoming_edges)
        )
        return self.layer2(hidden, graph.edge_index, graph.edge_type,incoming_edges=graph.incoming_edges)

    def action_logits(
        self,
        graph: BlockRobotGraph,
        embeddings: torch.Tensor,
        block_id: str,
        dynamic_rows: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        block_embedding = embeddings[graph.block_index[block_id]].expand(4, -1)
        robot_embeddings = embeddings[graph.robot_offset : graph.robot_offset + 4]
        logits = self.policy(
            torch.cat((block_embedding, robot_embeddings, dynamic_rows), dim=1)
        ).squeeze(1)
        return logits.masked_fill(~legal_mask, -torch.inf)

    def action_probabilities(
        self,
        graph: BlockRobotGraph,
        embeddings: torch.Tensor,
        block_id: str,
        dynamic_rows: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return P(robot | block, partial assignment, graph) on legal actions."""
        return torch.softmax(
            self.action_logits(
                graph, embeddings, block_id, dynamic_rows, legal_mask
            ),
            dim=0,
        )


class BlockAssignmentRankingModel(nn.Module):
    """Small joint assignment/ranking model used by the initial-solution study.

    ``encoder_kind='gnn'`` retains the frozen two-layer, four-head GAT.  The MLP
    ablation consumes the identical node and dynamic features but performs no
    message passing.  Weld-edge ablation is controlled when the graph is built,
    so it is used consistently during both training and inference.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        heads: int = 4,
        *,
        encoder_kind: str = "gnn",
    ) -> None:
        super().__init__()
        if encoder_kind not in {"gnn", "mlp"}:
            raise ValueError("encoder_kind must be 'gnn' or 'mlp'")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.encoder_kind = encoder_kind
        if encoder_kind == "gnn":
            self.layer1 = SparseGATLayer(NODE_DIM, hidden_dim, heads)
            self.layer2 = SparseGATLayer(hidden_dim, hidden_dim, heads)
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(NODE_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        fused_dim = 2 * hidden_dim + DYNAMIC_DIM
        self.assignment_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_static(self, graph: BlockRobotGraph) -> torch.Tensor:
        if self.encoder_kind == "mlp":
            return self.node_mlp(graph.node_features)
        hidden = torch.nn.functional.elu(
            self.layer1(graph.node_features, graph.edge_index, graph.edge_type,
                        incoming_edges=graph.incoming_edges)
        )
        return self.layer2(hidden, graph.edge_index, graph.edge_type,incoming_edges=graph.incoming_edges)

    def _rows(
        self,
        graph: BlockRobotGraph,
        embeddings: torch.Tensor,
        block_id: str,
        dynamic_rows: torch.Tensor,
    ) -> torch.Tensor:
        block_embedding = embeddings[graph.block_index[block_id]].expand(4, -1)
        robot_embeddings = embeddings[graph.robot_offset : graph.robot_offset + 4]
        return torch.cat((block_embedding, robot_embeddings, dynamic_rows), dim=1)

    def action_logits(
        self,
        graph: BlockRobotGraph,
        embeddings: torch.Tensor,
        block_id: str,
        dynamic_rows: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.assignment_head(
            self._rows(graph, embeddings, block_id, dynamic_rows)
        ).squeeze(1)
        return logits.masked_fill(~legal_mask, -torch.inf)

    def ordering_score(
        self,
        graph: BlockRobotGraph,
        embeddings: torch.Tensor,
        block_id: str,
        robot: int,
        dynamic_rows: torch.Tensor,
    ) -> torch.Tensor:
        if robot not in range(4):
            raise ValueError("robot must be 0..3")
        return self.ranking_head(
            self._rows(graph, embeddings, block_id, dynamic_rows)[robot]
        ).squeeze(0)


JOINT_DYNAMIC_DIM = 41
JOINT_PARENT_DIM = 12


class JointInsertionGNN(BlockAssignmentGNN):
    """The existing two-layer GAT with conditional parent/insertion heads."""

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=False, opportunity_features=False):
        super().__init__(hidden_dim, heads)
        self.hidden_dim, self.heads = hidden_dim, heads
        self.opportunity_features=opportunity_features
        self.joint_feature_dim=JOINT_DYNAMIC_DIM+8*int(opportunity_features)
        self.parent_feature_dim=JOINT_PARENT_DIM+8*int(opportunity_features)
        self.policy = nn.Identity()  # Assignment-only head is not used here.
        self.parent_head = nn.Sequential(nn.Linear(hidden_dim+self.parent_feature_dim,hidden_dim),
                                         nn.ReLU(),nn.Linear(hidden_dim,1))
        self.insertion_head = nn.Sequential(nn.Linear(4*hidden_dim+self.joint_feature_dim,hidden_dim),
                                            nn.ReLU(),nn.Linear(hidden_dim,1))
        # Optional TRAIN-fitted scaling; the default preserves V1 checkpoints
        # and logits exactly. No new direction head or encoder is introduced.
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            for prefix,dim in (('joint',self.joint_feature_dim),('parent',self.parent_feature_dim)):
                self.register_buffer(prefix+'_mean',torch.zeros(dim))
                self.register_buffer(prefix+'_scale',torch.ones(dim))

    def fit_input_scaling(self, steps):
        if not self.normalize_inputs:
            raise ValueError('input scaling is disabled')
        for prefix,key in (('joint','features'),('parent','parent_features')):
            rows=torch.cat([s[key] for s in steps if s[key] is not None and len(s[key])])
            if not torch.isfinite(rows).all():
                raise ValueError('nonfinite TRAIN features')
            getattr(self,prefix+'_mean').copy_(rows.mean(0))
            getattr(self,prefix+'_scale').copy_(rows.std(0,unbiased=False).clamp_min(1e-3))

    def fit_opportunity_scaling(self,steps):
        """Fit only the eight appended TRAIN features; retain legacy scalers."""
        if not self.opportunity_features or not self.normalize_inputs:
            raise ValueError('normalized opportunity inputs are required')
        for prefix,key in (('joint','features'),('parent','parent_features')):
            rows=torch.cat([s[key][:,-8:] for s in steps if s[key] is not None and len(s[key])])
            if not torch.isfinite(rows).all():raise ValueError('nonfinite TRAIN opportunities')
            getattr(self,prefix+'_mean')[-8:].copy_(rows.mean(0))
            getattr(self,prefix+'_scale')[-8:].copy_(rows.std(0,unbiased=False).clamp_min(1e-3))

    def scale_inputs(self, features, prefix):
        if self.normalize_inputs:
            return (features-getattr(self,prefix+'_mean'))/getattr(self,prefix+'_scale')
        return features

    def parent_logits(self, graph, embeddings, parent_ids, features):
        pooled = torch.stack([embeddings[graph.parent_block_indices[p]].mean(0) for p in parent_ids])
        return self.parent_head(torch.cat((pooled,self.scale_inputs(features,'parent')),dim=1)).squeeze(1)

    def joint_action_logits(self, embeddings, indices, features):
        context = embeddings[indices.clamp_min(0)] * (indices>=0).unsqueeze(-1)
        return self.insertion_head(torch.cat((context.flatten(1),self.scale_inputs(features,'joint')),dim=1)).squeeze(1)

    def contextualize_state(self, graph, embeddings, routes, directions):
        """Legacy policies retain their original static embeddings exactly."""
        return embeddings


class CoordinationJointInsertionGNN(JointInsertionGNN):
    """Refresh route/order/direction and same-rail context at every decision.

    These edges describe the student's current partial solution only. They do
    not assert schedulability: complete candidates still need B32 and the
    independent certifier. The original static policy/checkpoints are intact.
    """

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=True, opportunity_features=False):
        super().__init__(hidden_dim, heads, normalize_inputs=normalize_inputs,opportunity_features=opportunity_features)
        # Keep each weld's own geometry when attention neighborhoods coincide.
        self.geometry_skip = nn.Linear(NODE_DIM, hidden_dim)
        self.route_direction = nn.Embedding(2, hidden_dim)
        self.route_messages = nn.ModuleDict({
            relation: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for relation in ('successor', 'predecessor', 'owner', 'rail')
        })
        self.route_norm = nn.LayerNorm(hidden_dim)
        self.use_dynamic_context = True

    def encode_static(self, graph):
        return super().encode_static(graph) + self.geometry_skip(graph.node_features)

    def contextualize_state(self, graph, embeddings, routes, directions):
        if not self.use_dynamic_context:
            return embeddings
        if len(routes) != 4 or len(directions) != 4:
            raise ValueError('dynamic context expects four directed routes')
        edges = {name: [] for name in self.route_messages}
        placed, bits = [], []
        for robot, (route, ds) in enumerate(zip(routes, directions)):
            if len(route) != len(ds) or any(d not in (0, 1) for d in ds):
                raise ValueError('dynamic context has invalid route directions')
            nodes = [graph.block_index[b] for b in route]
            placed.extend(nodes)
            bits.extend(ds)
            owner = graph.robot_offset + robot
            for node in nodes:
                edges['owner'].extend(((node, owner), (owner, node)))
            edges['successor'].extend(zip(nodes[:-1], nodes[1:]))
            edges['predecessor'].extend(zip(nodes[1:], nodes[:-1]))
        if len(set(placed)) != len(placed):
            raise ValueError('dynamic context contains duplicate blocks')
        for robot in range(4):
            edges['rail'].append((graph.robot_offset + robot,
                                  graph.robot_offset + (robot ^ 1)))
        edge_tensors = {
            name: torch.tensor(pairs, dtype=torch.long, device=embeddings.device).t()
            for name, pairs in edges.items() if pairs
        }
        h = embeddings
        if placed:
            indices = torch.tensor(placed, dtype=torch.long, device=h.device)
            orientation = torch.tensor(bits, dtype=torch.long, device=h.device)
            h = h.index_add(0, indices, self.route_direction(orientation))
        # Ordered neighbors -> route owner -> same-rail partner is three hops.
        # Reuse the same transforms at each hop; no future route is supplied.
        for _ in range(3):
            message = torch.zeros_like(h)
            degree = h.new_zeros((len(h), 1))
            for relation, (source, destination) in edge_tensors.items():
                message = message.index_add(0, destination,
                                            self.route_messages[relation](h[source]))
                degree = degree.index_add(0, destination,
                                          h.new_ones((len(destination), 1)))
            h = self.route_norm(h + nn.functional.elu(message / degree.clamp_min(1)))
        return h


class ModeJointInsertionGNN(CoordinationJointInsertionGNN):
    """Choose legal parent modes before constructing their directed routes."""

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=True, opportunity_features=False):
        super().__init__(hidden_dim, heads, normalize_inputs=normalize_inputs,opportunity_features=opportunity_features)
        self.use_dynamic_context = False
        self.mode_head = nn.Sequential(nn.Linear(hidden_dim + 12, hidden_dim),
                                       nn.ReLU(), nn.Linear(hidden_dim, 1))

    def mode_logits(self, graph, embeddings, parent_ids, features):
        pooled = {parent: embeddings[graph.parent_block_indices[parent]].mean(0)
                  for parent in dict.fromkeys(parent_ids)}
        context = torch.stack([pooled[parent] for parent in parent_ids])
        return self.mode_head(torch.cat((context, features), dim=1)).squeeze(1)


def legal_mode_inputs(parents, config):
    """Static parent geometry and each currently legal mode, in catalog order."""
    parents = tuple(sorted(parents, key=lambda parent: parent.parent_id))
    graph, _, _ = joint_graph(joint_geometry(parents, fixed_split_patterns(parents, config), config), config)
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    candidates, features = [], []
    for parent in parents:
        for pattern in catalog[parent.parent_id]:
            children = blocks_for_pattern(parent, pattern, config)
            lengths = [child.length for child in children]
            process = [config.process_time(length) for length in lengths]
            candidates.append(pattern)
            features.append([float(pattern.kind is SplitKind.WHOLE),
                             float(pattern.kind is SplitKind.Y_SPLIT),
                             float(pattern.kind is SplitKind.X_SPLIT), float(pattern.mandatory),
                             pattern.t if pattern.t is not None else .5,
                             float(pattern.rail is Rail.UPPER), float(pattern.rail is Rail.LOWER),
                             len(children)/2, min(lengths)/parent.length, max(lengths)/parent.length,
                             min(process)/2500, max(process)/2500])
    return graph, tuple(candidates), torch.tensor(features, dtype=torch.float32).reshape(-1, 12)


def select_gnn_patterns(parents, model, config):
    """One deterministic argmax per parent; no route or teacher is an input."""
    graph, candidates, features = legal_mode_inputs(parents, config)
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode_static(graph)
        logits = model.mode_logits(graph, embeddings, tuple(p.parent_id for p in candidates), features)
        if not torch.isfinite(logits).all():
            raise ValueError('NONFINITE_MODE_LOGITS')
        best = {}
        for index, pattern in enumerate(candidates):
            previous = best.get(pattern.parent_id)
            if previous is None or logits[index] > logits[previous]:
                best[pattern.parent_id] = index
    return tuple(candidates[best[parent.parent_id]] for parent in sorted(parents,key=lambda p:p.parent_id))


@dataclass(frozen=True)
class JointCompletionResult:
    solution: CanonicalSolution | None
    directions: tuple[tuple[int, ...], ...] | None
    actions: tuple
    inference_seconds: float
    rejection_reason: str | None = None
    oriented_state: object | None = None


@dataclass(frozen=True)
class JointGeometry:
    """Geometry and fixed patterns only: deliberately has no ownership/routes."""
    parents: tuple[ParentWeld, ...]
    patterns: tuple[SplitPattern, ...]
    revision: int = 0


def joint_geometry(parents, patterns, config):
    parents=tuple(sorted(parents,key=lambda p:p.parent_id))
    patterns=tuple(sorted(patterns,key=lambda p:p.parent_id))
    ids=[p.parent_id for p in parents]
    if len(set(ids))!=len(ids) or len(patterns)!=len(parents) or {p.parent_id for p in patterns}!=set(ids):
        raise ValueError('one fixed pattern per distinct parent is required')
    geometry=JointGeometry(parents,patterns)
    # Existing formal pattern validation; no complete assignment is invented.
    block_map(geometry,config,scope=FORMAL_SCOPE_V2)
    return geometry


def joint_graph(reference, config):
    blocks = block_map(reference,config,scope=FORMAL_SCOPE_V2)
    graph = build_block_robot_graph(tuple(sorted(blocks.values(),key=lambda b:b.block_id)),
                                    reference.patterns,config,include_parent_edges=True)
    return graph, blocks, {p.parent_id:p for p in reference.patterns}


def joint_start_state(reference, routes, directions, pending, blocks, config):
    from .lns import oriented_route_state
    pending = set(pending)
    if not pending <= {p.parent_id for p in reference.parents}:
        raise ValueError('unknown pending parent')
    actual = [b for route in routes for b in route]
    expected = {b for b in blocks if blocks[b].parent_id not in pending}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('partial state must remove each pending parent completely')
    state = oriented_route_state(routes,directions,blocks,config)
    for r,route in enumerate(routes):
        if any(not robot_is_eligible(blocks[b],r,config) for b in route):
            raise ValueError('illegal retained assignment')
        patterns={p.parent_id:p for p in reference.patterns}
        for b in route:
            p=patterns[blocks[b].parent_id]
            if p.kind is SplitKind.X_SPLIT:
                required=(0 if p.rail is Rail.UPPER else 2)+int(b.endswith('::1'))
                if r!=required:
                    raise ValueError('retained X_SPLIT child has wrong robot')
    return state


def remaining_robot_work(state,blocks,patterns,config,*,insertion_limit=8,check_initial_feasibility=True):
    """Current legal insertion opportunities, not allocated load or a mask.

    Eligibility shares the exact existing reach/rail/mode/head/sibling query.
    A flexible child's processing time is counted for every eligible robot.
    Zero current options does not prune a future parent or force a fallback.
    """
    from .lns import has_joint_block_candidate
    placed={b for route in state.routes for b in route}
    eligible=[0.]*4;exclusive=[0.]*4
    for child in blocks.values():
        if child.block_id in placed:continue
        robots=[r for r in range(4) if robot_is_eligible(child,r,config) and
                has_joint_block_candidate(state,child,patterns[child.parent_id],blocks,config,
                    insertion_limit=insertion_limit,check_initial_feasibility=check_initial_feasibility,robot_id=r)]
        processing=config.process_time(child.length)
        for r in robots:eligible[r]+=processing
        if len(robots)==1:exclusive[robots[0]]+=processing
    scale=max(1.,sum(config.process_time(b.length) for b in blocks.values()))
    return tuple(v/scale for v in eligible+exclusive)


def joint_parent_inputs(state, pending, blocks, patterns, config, *, insertion_limit=8,check_initial_feasibility=True,
                        include_opportunities=False):
    from .lns import has_joint_block_candidate
    scale = max(1.0,sum(config.process_time(b.length) for b in blocks.values()))
    travel = [sum(arcs) for arcs in state.arc_costs]
    grouped = {}
    for block in blocks.values():
        grouped.setdefault(block.parent_id, []).append(block)
    prefix = [*(v/scale for v in state.loads), *(v/scale for v in travel)]
    opportunities=remaining_robot_work(state,blocks,patterns,config,insertion_limit=insertion_limit,
        check_initial_feasibility=check_initial_feasibility) if include_opportunities else ()
    ids, features = [], []
    for p in sorted(pending):
        children = sorted(grouped[p],key=lambda b:b.block_id)
        if not has_joint_block_candidate(state,children[0],patterns[p],blocks,config,
                                         insertion_limit=insertion_limit,
                                         check_initial_feasibility=check_initial_feasibility):
            continue
        legal = [r for r in range(4) if robot_is_eligible(children[0],r,config)]
        if patterns[p].kind is SplitKind.X_SPLIT:
            legal = [0 if patterns[p].rail is Rail.UPPER else 2]
        ids.append(p)
        features.append([*prefix,
                         sum(config.process_time(b.length) for b in children)/scale,
                         min(state.loads[r]+travel[r] for r in legal)/scale,
                         len(pending)/max(1,len(patterns)),len(children)/2,*opportunities])
    return tuple(ids),torch.tensor(features,dtype=torch.float32).reshape(-1,JOINT_PARENT_DIM+8*int(include_opportunities))


def joint_candidate_inputs(graph, state, variants, blocks, config, *, include_opportunities=False,patterns=None,
                           insertion_limit=8,check_initial_feasibility=True):
    """Only geometry, the current prefix and executable local next states."""
    from mrta_reference.geometry import oriented_endpoints
    scale = max(1.0,sum(config.process_time(b.length) for b in blocks.values()))
    n = max(1,len(blocks))
    old_travel = [sum(arcs) for arcs in state.arc_costs]
    prefix = [*(v/scale for v in state.loads), *(v/scale for v in old_travel)]
    opportunities=remaining_robot_work(state,blocks,patterns,config,insertion_limit=insertion_limit,
        check_initial_feasibility=check_initial_feasibility) if include_opportunities else ()
    route_lengths = [len(route)/n for route in state.routes]
    # A position's eight direction variants share identical route geometry.
    # These local caches retain the original sums and min/max operations.
    bounds = {}
    arc_sums = {id(arcs): value for arcs, value in zip(state.arc_costs, old_travel)}
    endpoints = {}
    def oriented(block_id, direction):
        key = (block_id, direction)
        if key not in endpoints:
            endpoints[key] = oriented_endpoints(blocks[block_id], direction)
        return endpoints[key]
    def route_bounds(route):
        if route not in bounds:
            bounds[route] = (max(max(blocks[b].start[0],blocks[b].end[0]) for b in route),
                             min(min(blocks[b].start[0],blocks[b].end[0]) for b in route))
        return bounds[route]
    def arc_sum(arcs):
        key = id(arcs)
        if key not in arc_sums:
            arc_sums[key] = sum(arcs)
        return arc_sums[key]
    def overlap(s,r):
        left,right = (r//2*2,r//2*2+1)
        if not s.routes[left] or not s.routes[right]:
            return 0.0
        left_end = route_bounds(s.routes[left])[0]
        right_start = route_bounds(s.routes[right])[1]
        return max(0.0,left_end+config.interference_dx-right_start)/20
    def first_x(s,r):
        return (oriented(s.routes[r][0],s.directions[r][0])[0][0]/20
                if s.routes[r] else 0.0)
    old_overlap = [overlap(state,r) for r in range(4)]
    old_first_x = [first_x(state,r) for r in range(4)]
    indices, features = [], []
    for after,action in variants:
        r,k = action.robot,action.position
        route = state.routes[r]
        pre = route[k-1] if k else None
        post = route[k] if k<len(route) else None
        indices.append([graph.block_index[action.block_id],graph.robot_offset+r,
                        graph.block_index[pre] if pre else -1,graph.block_index[post] if post else -1])
        start,end = oriented(action.block_id,action.direction)
        pre_old = state.directions[r][k-1] if pre else 0
        post_old = state.directions[r][k] if post else 0
        pre_new = after.directions[r][k-1] if pre else 0
        post_new = after.directions[r][k+1] if post else 0
        incoming = (math.dist(oriented(pre,pre_new)[1],start)/config.empty_speed if pre else 0)
        outgoing = (math.dist(end,oriented(post,post_new)[0])/config.empty_speed if post else 0)
        old_arc = (math.dist(oriented(pre,pre_old)[1],oriented(post,post_old)[0])/config.empty_speed if pre and post else 0)
        estimated = [p+arc_sum(arcs) for p,arcs in zip(after.loads,after.arc_costs)]
        features.append([*prefix,
                         *(v/scale for v in estimated),*route_lengths,
                         k/max(1,len(route)),float(pre is not None),float(post is not None),float(action.direction),
                         pre_old,pre_new,post_old,post_new,len(action.neighbour_changes)/2,
                         incoming/scale,outgoing/scale,old_arc/scale,(after.travel-state.travel)/scale,
                         old_overlap[r],overlap(after,r),float(r%2==0),
                         start[0]/20,start[1]/12,end[0]/20,end[1]/12,
                         old_first_x[r],first_x(after,r),old_first_x[r^1],max(estimated)/scale,sum(estimated)/scale,*opportunities])
    return torch.tensor(indices,dtype=torch.long).reshape(-1,4),torch.tensor(features,dtype=torch.float32).reshape(-1,JOINT_DYNAMIC_DIM+8*int(include_opportunities))


def prepare_joint_teacher_steps(reference, routes, directions, pending, actions, config, *, insertion_limit=8,include_opportunities=False):
    """Validate every teacher label against the inference action set; never skip."""
    from .lns import joint_block_candidates,joint_action_key,OrientedInsertionAction
    graph,blocks,patterns = joint_graph(reference,config)
    state = joint_start_state(reference,routes,directions,pending,blocks,config)
    pending = set(pending)
    steps = []
    cursor = 0
    while pending:
        if cursor>=len(actions):
            raise ValueError('teacher trajectory ended before complete parent coverage')
        raw = actions[cursor]
        parent = blocks[raw['block_id']].parent_id
        ids,parent_features = joint_parent_inputs(state,pending,blocks,patterns,config,insertion_limit=insertion_limit,
                                                  include_opportunities=include_opportunities)
        if parent not in ids:
            raise ValueError(f'teacher parent absent from legal candidates: {parent}')
        children = sorted((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
        for child_index,child in enumerate(children):
            if cursor>=len(actions) or actions[cursor]['block_id']!=child.block_id:
                raise ValueError('teacher must place both children atomically in block order')
            raw = actions[cursor]
            action = OrientedInsertionAction(raw['block_id'],raw['robot'],raw['position'],raw['direction'],
                                              tuple(tuple(c) for c in raw['neighbour_changes']))
            variants = joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=insertion_limit)
            matches = [i for i,(_,a) in enumerate(variants) if joint_action_key(a)==joint_action_key(action)]
            if len(matches)!=1:
                raise ValueError(f'teacher action missing/ambiguous: {joint_action_key(action)}; candidates={len(variants)}')
            indices,features = joint_candidate_inputs(graph,state,variants,blocks,config,
                include_opportunities=include_opportunities,patterns=patterns,insertion_limit=insertion_limit)
            steps.append(dict(parent_ids=ids if child_index==0 else (),parent_features=parent_features if child_index==0 else None,
                              parent_target=ids.index(parent) if child_index==0 else None,
                              indices=indices,features=features,target=matches[0],
                              candidate_actions=tuple(a for _,a in variants),
                              state_routes=state.routes,state_directions=state.directions,
                              pending_parent_ids=tuple(sorted(pending)),teacher_action_key=joint_action_key(action)))
            state = variants[matches[0]][0]
            cursor += 1
        pending.remove(parent)
    if cursor!=len(actions):
        raise ValueError('teacher trajectory contains extra actions')
    return graph,steps,state


class _JointActionUnavailable(ValueError):
    """An ordinary policy dead end, distinct from corrupt inputs/numerics."""


@dataclass(frozen=True)
class JointTrainingRollout:
    completion: JointCompletionResult
    parent_log_probs: tuple[torch.Tensor, ...]
    insertion_log_probs: tuple[torch.Tensor, ...]
    log_prob: torch.Tensor


def mrv_parent(state, parent_ids, blocks, patterns, config, *, insertion_limit=8):
    """Cheap robot/position count product, then LPT and stable parent ID.

    parent_ids is the same feasible-first-child pool used by the learned head.
    Counts use current route lengths for every child, without enumerating joint
    combinations or changing any direction/start/sibling candidate pruning.
    """
    from .initialization import bounded_insertion_positions
    def priority(parent):
        children=sorted((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
        choices=1
        for child in children:
            robots=[r for r in range(4) if robot_is_eligible(child,r,config)]
            if patterns[parent].kind is SplitKind.X_SPLIT:
                robots=[(0 if patterns[parent].rail is Rail.UPPER else 2)+int(child.block_id.endswith('::1'))]
            choices*=sum(len(bounded_insertion_positions(len(state.routes[r]),insertion_limit)) for r in robots)
        return choices,-sum(config.process_time(b.length) for b in children),parent
    return min(parent_ids,key=priority)


def rule_action_costs(variants,preferred_owners):
    """The existing soft ownership and unrounded P+MOVE ordering."""
    costs=[]
    for after,action in variants:
        completion=tuple(p+sum(arcs) for p,arcs in zip(after.loads,after.arc_costs))
        costs.append((int(action.robot!=preferred_owners[action.block_id]),max(completion),sum(completion)))
    return costs


def rule_guided_action_logits(logits, variants, preferred_owners):
    """Parameter-free rank fusion over the unchanged legal insertion pool.

    A rule owner is only a preference. Every supplied action receives a score;
    rank fusion neither removes actions nor fixes a robot or direction. The
    existing greedy/sample decoder operates on the resulting negative ranks.
    """
    values=logits.detach().cpu().tolist()
    costs=rule_action_costs(variants,preferred_owners)
    neural_order=sorted(range(len(values)),key=lambda i:(-values[i],i))
    rule_order=sorted(range(len(values)),key=lambda i:(costs[i],i))
    neural_rank={i:r for r,i in enumerate(neural_order)}
    rule_rank={i:r for r,i in enumerate(rule_order)}
    fused_order=sorted(range(len(values)),key=lambda i:(neural_rank[i]+rule_rank[i],costs[i],-values[i],i))
    fused_rank={i:r for r,i in enumerate(fused_order)}
    return logits.new_tensor([-fused_rank[i] for i in range(len(values))])


def rule_anchored_action_logits(logits,variants,preferred_owners):
    """Freeze the rule cost ordering; let the GNN distinguish equal-cost actions.

    This is a frozen-policy diagnostic, not a learned Cmax residual estimator.
    All actions retain finite ranks and remain available to sampling. Costs use
    the same unrounded values as rank fusion; no tolerance or weight is added.
    """
    costs=rule_action_costs(variants,preferred_owners)
    values=logits.detach().cpu().tolist()
    order=sorted(range(len(values)),key=lambda i:(costs[i],-values[i],i))
    ranks={i:r for r,i in enumerate(order)}
    return logits.new_tensor([-ranks[i] for i in range(len(values))])


def _decode_joint(reference, routes, directions, pending_parent_ids, model, config,
                  *, insertion_limit, choose,allow_partial_start_conflicts=False,prefer_legal_start_recovery=True,
                  parent_selection_mode='learned',action_selection_mode='gnn',action_observer=None,
                  forced_action=None,parent_observer=None,joint_greedy=False):
    """Shared state transitions and action space for inference and training."""
    from .lns import joint_block_candidates
    from .direction import _initial_error
    from mrta_reference.geometry import oriented_endpoints
    if parent_selection_mode not in ('learned','mrv'):
        raise ValueError('parent_selection_mode must be learned or mrv')
    if action_selection_mode not in ('gnn','gnn_rank','rule_guided','rule','rule_anchored'):
        raise ValueError('unknown action_selection_mode')
    started = time.perf_counter()
    actions = []
    try:
        graph,blocks,patterns = joint_graph(reference,config)
        preferred_owners=dict(rule_assignment(graph.blocks,config)) if action_selection_mode in ('rule','rule_guided','rule_anchored') else None
        state = joint_start_state(reference,routes,directions,pending_parent_ids,blocks,config)
        pending = set(pending_parent_ids)
        embeddings = model.encode_static(graph)
        opportunities=getattr(model,'opportunity_features',False)
        def valid_start(s):
            points={r:oriented_endpoints(blocks[route[0]],s.directions[r][0])[0]
                    for r,route in enumerate(s.routes) if route}
            return _initial_error(points,config) is None
        def check_start(s):
            # A destroyed state's exposed heads are not its final deployment.
            # Once a legal partial start is restored, retain the original prune.
            return not allow_partial_start_conflicts or valid_start(s)
        while pending:
            if parent_observer is not None:parent_observer(state,tuple(sorted(pending)),tuple(actions))
            ids,features = joint_parent_inputs(state,pending,blocks,patterns,config,insertion_limit=insertion_limit,
                                               check_initial_feasibility=check_start(state),include_opportunities=opportunities)
            if not ids:
                raise _JointActionUnavailable('NO_LEGAL_PARENT_ACTION')
            forcing=forced_action is not None
            joint_first=None
            if forcing:
                parent=blocks[forced_action.block_id].parent_id
                if parent not in ids:raise ValueError('forced parent is not a current legal action')
                current_embeddings=model.contextualize_state(graph,embeddings,state.routes,state.directions)
            elif parent_selection_mode=='learned':
                current_embeddings = model.contextualize_state(graph,embeddings,state.routes,state.directions)
                parent_logits = model.parent_logits(graph,current_embeddings,ids,features)
                if not torch.isfinite(parent_logits).all():
                    raise ValueError('NONFINITE_PARENT_LOGITS')
                if joint_greedy:
                    # Normalize over ALL currently legal parents. Each of the
                    # two examined parents retains its own full conditional
                    # insertion pool, including neighbour direction changes.
                    parent_logp=torch.log_softmax(parent_logits,dim=0)
                    order=sorted(range(len(ids)),key=lambda i:(-float(parent_logits[i]),i))[:2]
                    best_score=None
                    for index in order:
                        candidate_parent=ids[index]
                        child=min((b for b in blocks.values() if b.parent_id==candidate_parent),key=lambda b:b.block_id)
                        variants=joint_block_candidates(state,child,patterns[candidate_parent],blocks,config,
                            insertion_limit=insertion_limit,check_initial_feasibility=check_start(state),
                            prefer_legal_start_recovery=prefer_legal_start_recovery)
                        if not variants:continue
                        indices,dynamic=joint_candidate_inputs(graph,state,variants,blocks,config,
                            include_opportunities=opportunities,patterns=patterns,insertion_limit=insertion_limit,
                            check_initial_feasibility=check_start(state))
                        logits=model.joint_action_logits(current_embeddings,indices,dynamic)
                        if not torch.isfinite(logits).all():raise ValueError('NONFINITE_INSERTION_LOGITS')
                        conditional=torch.log_softmax(logits,dim=0)
                        selected=int(torch.argmax(conditional))
                        score=float(parent_logp[index]+conditional[selected])
                        if best_score is None or score>best_score:
                            best_score=score
                            joint_first=(candidate_parent,variants,logits,selected)
                    if joint_first is None:raise _JointActionUnavailable('NO_LEGAL_CHILD_INSERTION')
                    parent=joint_first[0]
                else:
                    parent = ids[choose(parent_logits,'parent')]
            else:
                parent = mrv_parent(state,ids,blocks,patterns,config,insertion_limit=insertion_limit)
            children = sorted((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
            for child_index,child in enumerate(children):
                if child_index==0 and joint_first is not None:
                    _,variants,logits,selected=joint_first
                    if action_observer is not None:
                        action_observer(graph,state,variants,logits,selected,preferred_owners)
                    state,action=variants[selected]
                    actions.append(action)
                    continue
                variants = joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=insertion_limit,
                                                   check_initial_feasibility=check_start(state),
                                                   prefer_legal_start_recovery=prefer_legal_start_recovery)
                if not variants:
                    raise _JointActionUnavailable('NO_LEGAL_CHILD_INSERTION')
                indices,dynamic = joint_candidate_inputs(graph,state,variants,blocks,config,
                    include_opportunities=opportunities,patterns=patterns,insertion_limit=insertion_limit,
                    check_initial_feasibility=check_start(state))
                # The parent decision and its first child see the identical
                # state. Reuse only without gradients; training keeps its
                # original independent autograd accumulation paths.
                if child_index or parent_selection_mode!='learned' or torch.is_grad_enabled():
                    current_embeddings = model.contextualize_state(graph,embeddings,state.routes,state.directions)
                logits = model.joint_action_logits(current_embeddings,indices,dynamic)
                if not torch.isfinite(logits).all():
                    raise ValueError('NONFINITE_INSERTION_LOGITS')
                raw_logits=logits
                if action_selection_mode=='gnn_rank':
                    values=logits.detach().cpu().tolist()
                    order=sorted(range(len(variants)),key=lambda i:(-values[i],i))
                    ranks={i:r for r,i in enumerate(order)}
                    logits=logits.new_tensor([-ranks[i] for i in range(len(variants))])
                elif action_selection_mode=='rule':
                    costs=rule_action_costs(variants,preferred_owners)
                    order=sorted(range(len(variants)),key=lambda i:(costs[i],i))
                    ranks={i:r for r,i in enumerate(order)}
                    logits=logits.new_tensor([-ranks[i] for i in range(len(variants))])
                elif action_selection_mode=='rule_anchored':
                    logits=rule_anchored_action_logits(logits,variants,preferred_owners)
                elif preferred_owners is not None:
                    logits=rule_guided_action_logits(logits,variants,preferred_owners)
                if forcing and child_index==0:
                    from .lns import joint_action_key
                    matching=[i for i,(_,a) in enumerate(variants) if joint_action_key(a)==joint_action_key(forced_action)]
                    if len(matching)!=1:raise ValueError('forced first-child action is not in the unchanged legal pool')
                    selected=matching[0]
                    forced_action=None
                else:selected=choose(logits,'insertion')
                if action_observer is not None:
                    action_observer(graph,state,variants,raw_logits,selected,preferred_owners)
                state,action = variants[selected]
                actions.append(action)
            pending.remove(parent)
        if allow_partial_start_conflicts and not valid_start(state):
            raise _JointActionUnavailable('UNRESOLVED_PARTIAL_START_CONFLICT')
        solution = canonicalize(reference.parents,reference.patterns,
                                tuple(Route(r,state.routes[r]) for r in range(4)),config,scope=FORMAL_SCOPE_V2,
                                revision=reference.revision+1)
        if solution.patterns!=reference.patterns or tuple(r.block_ids for r in solution.routes)!=state.routes:
            # A completed policy path can make retained optional siblings
            # adjacent. Canonicalization then merges them into WHOLE, for which
            # the old child directions are no longer a valid explicit vector.
            # Reject this ordinary action-path dead end, as the rule repairer
            # does; do not abort the entire solver or silently reuse directions.
            raise _JointActionUnavailable('FIXED_PATTERN_CHANGED')
        return JointCompletionResult(solution,state.directions,tuple(actions),time.perf_counter()-started,
                                     oriented_state=state)
    except _JointActionUnavailable as error:
        return JointCompletionResult(None,None,tuple(actions),time.perf_counter()-started,str(error))


def construct_or_repair_with_gnn(reference, routes, directions, pending_parent_ids,
                                 model: JointInsertionGNN, config, *, insertion_limit=8,
                                 decode_mode='greedy', seed=0,allow_partial_start_conflicts=False,prefer_legal_start_recovery=True,
                                 parent_selection_mode='learned',action_selection_mode='gnn',action_observer=None,
                                 forced_action=None,parent_observer=None):
    """No-gradient decoding; complete results are scheduled outside.

    The opt-in partial-start repair can restore heads exposed by destruction.
    It retains the normal prune as soon as the partial start is legal and never
    returns an unresolved conflicting full start. Default decoding and training
    retain their original action space. When restoration is available, prefer
    it to staying in conflict; the preference can be disabled for ablation.
    parent_selection_mode changes only parent ordering: learned head or cheap
    deterministic MRV-LPT. Both retain the same insertion head and action pool.
    action_selection_mode='rule_guided' fuses neural/rule ranks at inference;
    the default gnn path and differentiable training rollout stay unchanged.
    'rule' and 'gnn_rank' are controlled ranking ablations using the same
    negative ordinal sampling scale and the complete original candidate pool.
    'rule_anchored' tests neural tie-breaking within the original rule costs.
    action_observer reads a pool before transition; it does not choose actions.
    'joint_greedy' adds finite lookahead over the top two learned parents and
    maximizes log P(parent) + log P(first-child insertion). It changes the
    search strategy; subsequent split children still use legal greedy insertion.
    """
    if decode_mode not in ('greedy','sample','joint_greedy'):
        raise ValueError('decode_mode must be greedy, sample or joint_greedy')
    if decode_mode=='joint_greedy' and (parent_selection_mode!='learned' or action_selection_mode!='gnn'):
        raise ValueError('joint_greedy requires the learned parent and GNN insertion heads')
    generator=None
    if decode_mode=='sample':
        generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    def choose(logits,kind):
        if decode_mode in ('greedy','joint_greedy'):
            return int(torch.argmax(logits))
        return int(torch.multinomial(torch.softmax(logits,dim=0),1,generator=generator).item())
    model.eval()
    with torch.inference_mode():
        return _decode_joint(reference,routes,directions,pending_parent_ids,model,config,
                             insertion_limit=insertion_limit,choose=choose,
                             allow_partial_start_conflicts=allow_partial_start_conflicts,
                             prefer_legal_start_recovery=prefer_legal_start_recovery,
                             parent_selection_mode=parent_selection_mode,action_selection_mode=action_selection_mode,
                             action_observer=action_observer,forced_action=forced_action,parent_observer=parent_observer,
                             joint_greedy=decode_mode=='joint_greedy')


def rollout_joint_repair(reference, routes, directions, pending_parent_ids,
                         model: JointInsertionGNN, config, *, insertion_limit=8, seed=0):
    """Differentiable temperature-1 sampling of the SAME executable actions.

    Only sampled indices are discrete. Both heads' selected log-softmax values
    retain the graph through the GAT. Numeric/data errors propagate to the caller;
    a genuine missing legal action preserves the sampled prefix for failure return.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('training rollout cannot run inside inference_mode')
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    logs={'parent':[],'insertion':[]}
    def choose(logits,kind):
        index=int(torch.multinomial(torch.softmax(logits,dim=0),1,generator=generator).item())
        logs[kind].append(torch.log_softmax(logits,dim=0)[index])
        return index
    # There is no stochastic dropout in this policy. Eval also makes the
    # self-critical baseline and sampled branch use the same model semantics.
    model.eval()
    with torch.enable_grad():
        completion=_decode_joint(reference,routes,directions,pending_parent_ids,model,config,
                                 insertion_limit=insertion_limit,choose=choose)
        terms=logs['parent']+logs['insertion']
        log_prob=torch.stack(terms).sum() if terms else next(model.parameters()).new_zeros(())
    return JointTrainingRollout(completion,tuple(logs['parent']),tuple(logs['insertion']),log_prob)


def _home(robot: int, config: ScientificConfig) -> tuple[float, float]:
    return (
        config.workspace_x[0] if robot in (0, 2) else config.workspace_x[1],
        config.by[1] if robot < 2 else config.by[0],
    )


def dynamic_features(
    block: WeldingBlock,
    assigned: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    total_process = max(
        1.0, sum(config.process_time(item.length) for item in blocks.values())
    )
    loads = [
        sum(config.process_time(blocks[block_id].length) for block_id in route)
        for route in assigned
    ]
    rail_loads = (loads[0] + loads[1], loads[2] + loads[3])
    mx = (block.start[0] + block.end[0]) / 2.0
    rows: list[list[float]] = []
    for robot in range(4):
        route = assigned[robot]
        last = (
            _home(robot, config)
            if not route
            else (
                (blocks[route[-1]].start[0] + blocks[route[-1]].end[0]) / 2.0,
                (blocks[route[-1]].start[1] + blocks[route[-1]].end[1]) / 2.0,
            )
        )
        xs = [
            value
            for block_id in route
            for value in (blocks[block_id].start[0], blocks[block_id].end[0])
        ]
        rail = 0 if robot < 2 else 1
        partner = robot + 1 if robot % 2 == 0 else robot - 1
        side_distance = mx / 20.0 if robot % 2 == 0 else (20.0 - mx) / 20.0
        rows.append(
            [
                loads[robot] / total_process,
                len(route) / max(1, total_steps),
                rail_loads[rail] / total_process,
                (rail_loads[0] - rail_loads[1]) / total_process,
                loads[partner] / total_process,
                last[0] / 20.0,
                last[1] / 12.0,
                (min(xs) if xs else last[0]) / 20.0,
                (max(xs) if xs else last[0]) / 20.0,
                math.dist(last, (mx, (block.start[1] + block.end[1]) / 2.0)) / 24.0,
                (loads[robot] + config.process_time(block.length)) / total_process,
                side_distance,
                step / max(1, total_steps - 1),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def rule_assignment(
    blocks: Sequence[WeldingBlock], config: ScientificConfig
) -> tuple[tuple[str, int], ...]:
    """One deterministic WAG-style rail balance plus spatial boundary assignment."""
    rail_groups: list[list[WeldingBlock]] = [[], []]
    flexible: list[WeldingBlock] = []
    for block in blocks:
        eligible = _eligible_robots(block, config)
        upper = any(robot < 2 for robot in eligible)
        lower = any(robot >= 2 for robot in eligible)
        if upper and lower:
            flexible.append(block)
        else:
            rail_groups[0 if upper else 1].append(block)
    rail_loads = [
        sum(config.process_time(block.length) for block in group)
        for group in rail_groups
    ]
    for block in sorted(
        flexible,
        key=lambda item: (-config.process_time(item.length), item.block_id),
    ):
        rail = min(range(2), key=lambda index: (rail_loads[index], index))
        rail_groups[rail].append(block)
        rail_loads[rail] += config.process_time(block.length)

    owner: dict[str, int] = {}
    for rail, group in enumerate(rail_groups):
        ordered = sorted(
            group,
            key=lambda item: (
                (item.start[0] + item.end[0]) / 2.0,
                item.block_id,
            ),
        )
        best = None
        for boundary in range(len(ordered) + 1):
            left = sum(config.process_time(item.length) for item in ordered[:boundary])
            right = sum(config.process_time(item.length) for item in ordered[boundary:])
            candidate = (max(left, right), abs(left - right), boundary)
            if best is None or candidate < best:
                best = candidate
        boundary = best[2]  # type: ignore[index]
        left_robot, right_robot = ((0, 1) if rail == 0 else (2, 3))
        for block in ordered[:boundary]:
            owner[block.block_id] = left_robot
        for block in ordered[boundary:]:
            owner[block.block_id] = right_robot
    return tuple((block.block_id, owner[block.block_id]) for block in blocks)


def solution_from_assignment(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    blocks: Sequence[WeldingBlock],
    assignment: Sequence[tuple[str, int]],
    config: ScientificConfig,
) -> CanonicalSolution:
    by_id = {block.block_id: block for block in blocks}
    assigned: list[list[str]] = [[] for _ in range(4)]
    seen: set[str] = set()
    for block_id, robot in assignment:
        if block_id in seen or block_id not in by_id:
            raise ValueError(f"duplicate or unknown assigned block: {block_id}")
        if not robot_is_eligible(by_id[block_id], robot, config):
            raise ValueError(f"illegal assignment {block_id} -> R{robot}")
        seen.add(block_id)
        assigned[robot].append(block_id)
    if seen != set(by_id):
        raise ValueError(f"assignment coverage mismatch: missing={sorted(set(by_id) - seen)}")
    # One common deterministic same-direction spatial order.  Both adjacent
    # robots progress in increasing X, which mirrors the simple WAG boundary
    # construction and reduces opposing-route conflicts without a route pool.
    routes = tuple(
        Route(
            robot,
            tuple(
                sorted(
                    assigned[robot],
                    key=lambda block_id: (
                        (by_id[block_id].start[0] + by_id[block_id].end[0]) / 2.0,
                        block_id,
                    ),
                )
            ),
        )
        for robot in range(4)
    )
    return canonicalize(parents, patterns, routes, config, scope=FORMAL_SCOPE_V2)


def solution_from_ordered_routes(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    blocks: Sequence[WeldingBlock],
    routes: Sequence[Sequence[str]],
    config: ScientificConfig,
) -> CanonicalSolution:
    """Validate ordered routes before passing them to the canonical model."""
    if len(routes) != 4:
        raise ValueError("exactly four robot routes are required")
    by_id = {block.block_id: block for block in blocks}
    actual = [block_id for route in routes for block_id in route]
    if len(actual) != len(set(actual)):
        raise ValueError("a ranked welding block is assigned more than once")
    if set(actual) != set(by_id):
        raise ValueError(
            "ranked route coverage mismatch: "
            f"missing={sorted(set(by_id) - set(actual))}, "
            f"extra={sorted(set(actual) - set(by_id))}"
        )
    for robot, route in enumerate(routes):
        for block_id in route:
            if not robot_is_eligible(by_id[block_id], robot, config):
                raise ValueError(f"illegal ranked assignment {block_id} -> R{robot}")
    return canonicalize(
        parents,
        patterns,
        tuple(Route(robot, tuple(route)) for robot, route in enumerate(routes)),
        config,
        scope=FORMAL_SCOPE_V2,
    )


def _validate_predecessors(
    block_ids: set[str],
    predecessors: Mapping[str, Sequence[str]] | None,
) -> dict[str, set[str]]:
    result = {block_id: set() for block_id in block_ids}
    for block_id, required in (predecessors or {}).items():
        if block_id not in block_ids:
            raise ValueError(f"unknown precedence target: {block_id}")
        result[block_id] = set(required)
        unknown = result[block_id] - block_ids
        if unknown:
            raise ValueError(f"unknown predecessors for {block_id}: {sorted(unknown)}")
        if block_id in result[block_id]:
            raise ValueError(f"self precedence for {block_id}")
    return result


def dag_safe_ordered_routes(
    model: BlockAssignmentRankingModel,
    graph: BlockRobotGraph,
    embeddings: torch.Tensor,
    assignment: Sequence[tuple[str, int]],
    config: ScientificConfig,
    *,
    predecessors: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Construct four routes through one global topological append sequence.

    The fixed WHOLE/mandatory-Y representation has coverage children but no
    parent-child temporal edge: the two mandatory-Y children are allowed to run
    in parallel on different rails.  ``predecessors`` therefore defaults to the
    empty relation, while keeping the constructor safe for any explicit frozen
    precedence relation supplied by a caller.  Every route edge and every
    supplied precedence edge follows the returned global append order, so their
    union cannot contain a cross-robot cycle.
    """
    by_id = {block.block_id: block for block in graph.blocks}
    owners: dict[str, int] = {}
    for block_id, robot in assignment:
        if block_id in owners or block_id not in by_id:
            raise ValueError(f"duplicate or unknown assigned block: {block_id}")
        if not robot_is_eligible(by_id[block_id], robot, config):
            raise ValueError(f"illegal assignment {block_id} -> R{robot}")
        owners[block_id] = robot
    if set(owners) != set(by_id):
        raise ValueError("assignment does not cover the graph blocks")
    required = _validate_predecessors(set(by_id), predecessors)
    remaining = set(by_id)
    scheduled: set[str] = set()
    routes: list[list[str]] = [[] for _ in range(4)]
    step = 0
    while remaining:
        ready = {
            block_id
            for block_id in remaining
            if required[block_id].issubset(scheduled)
        }
        if not ready:
            unresolved = {key: sorted(required[key] - scheduled) for key in sorted(remaining)}
            raise ValueError(f"precedence graph contains a cycle: {unresolved}")

        chosen: str | None = None
        # Rotate the robot turn so ranking comparisons remain within one robot;
        # the expert data provide no meaningful cross-robot ordering labels.
        for offset in range(4):
            robot = (step + offset) % 4
            candidates = sorted(block_id for block_id in ready if owners[block_id] == robot)
            if not candidates:
                continue
            scored: list[tuple[float, str]] = []
            for block_id in candidates:
                rows = dynamic_features(
                    by_id[block_id], routes, by_id, config, step, len(by_id)
                )
                score = model.ordering_score(
                    graph, embeddings, block_id, robot, rows
                )
                scored.append((float(score.detach()), block_id))
            chosen = min(scored, key=lambda item: (-item[0], item[1]))[1]
            break
        if chosen is None:
            raise AssertionError("ready tasks have no assigned robot")
        routes[owners[chosen]].append(chosen)
        remaining.remove(chosen)
        scheduled.add(chosen)
        step += 1
    return tuple(tuple(route) for route in routes)


def construct_ranked_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    model: BlockAssignmentRankingModel,
    *,
    assignment_kind: str,
    include_weld_edges: bool = True,
    predecessors: Mapping[str, Sequence[str]] | None = None,
) -> ConstructionResult:
    """Create a direct fixed-split solution with learned within-robot ordering."""
    if assignment_kind not in {"RULE", "MODEL"}:
        raise ValueError("assignment_kind must be RULE or MODEL")
    started = time.perf_counter()
    patterns = fixed_split_patterns(parents, config)
    blocks = fixed_blocks(parents, patterns, config)
    graph = build_block_robot_graph(
        blocks, patterns, config, include_weld_edges=include_weld_edges
    )
    by_id = {block.block_id: block for block in blocks}
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode_static(graph)
        if assignment_kind == "RULE":
            assignment = rule_assignment(blocks, config)
        else:
            partial: list[list[str]] = [[] for _ in range(4)]
            rows: list[tuple[str, int]] = []
            for step, block in enumerate(blocks):
                mask = torch.tensor(
                    [robot_is_eligible(block, robot, config) for robot in range(4)],
                    dtype=torch.bool,
                )
                logits = model.action_logits(
                    graph,
                    embeddings,
                    block.block_id,
                    dynamic_features(block, partial, by_id, config, step, len(blocks)),
                    mask,
                )
                robot = int(torch.argmax(logits).item())
                partial[robot].append(block.block_id)
                rows.append((block.block_id, robot))
            assignment = tuple(rows)
        routes = dag_safe_ordered_routes(
            model,
            graph,
            embeddings,
            assignment,
            config,
            predecessors=predecessors,
        )
    solution = solution_from_ordered_routes(
        parents, patterns, blocks, routes, config
    )
    elapsed = time.perf_counter() - started
    return ConstructionResult(solution, elapsed, 1.0, tuple(assignment))


def construct_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    *,
    model: BlockAssignmentGNN | None = None,
) -> ConstructionResult:
    started = time.perf_counter()
    patterns = fixed_split_patterns(parents, config)
    blocks = fixed_blocks(parents, patterns, config)
    if model is None:
        assignment = rule_assignment(blocks, config)
        elapsed = time.perf_counter() - started
        return ConstructionResult(
            solution_from_assignment(parents, patterns, blocks, assignment, config),
            elapsed,
            1.0,
            assignment,
        )

    graph = build_block_robot_graph(blocks, patterns, config)
    assigned: list[list[str]] = [[] for _ in range(4)]
    by_id = {block.block_id: block for block in blocks}
    assignment_rows: list[tuple[str, int]] = []
    legal = 0
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode_static(graph)
        for step, block in enumerate(blocks):
            mask = torch.tensor(
                [robot_is_eligible(block, robot, config) for robot in range(4)],
                dtype=torch.bool,
            )
            probabilities = model.action_probabilities(
                graph,
                embeddings,
                block.block_id,
                dynamic_features(block, assigned, by_id, config, step, len(blocks)),
                mask,
            )
            robot = int(torch.argmax(probabilities).item())
            legal += int(bool(mask[robot]))
            assigned[robot].append(block.block_id)
            assignment_rows.append((block.block_id, robot))
    elapsed = time.perf_counter() - started
    assignment = tuple(assignment_rows)
    return ConstructionResult(
        solution_from_assignment(parents, patterns, blocks, assignment, config),
        elapsed,
        legal / max(1, len(blocks)),
        assignment,
    )


def assignment_targets(solution: CanonicalSolution) -> dict[str, int]:
    return {
        block_id: route.robot_id
        for route in solution.routes
        for block_id in route.block_ids
    }

