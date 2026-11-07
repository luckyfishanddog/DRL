"""GNN welding assignment and complete-parent route construction.

The fixed-pattern policies and the joint all-pattern constructor share the existing
two-layer GAT and native legal insertion states. CompleteParentGNN scores mode,
robot assignment, insertion positions and directions as a single atomic action.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import product
import math
import time
from typing import Mapping, NamedTuple, Sequence

import torch
from torch import nn

from mrta_ranker.model import SparseGATLayer
from mrta_reference.geometry import (
    blocks_for_pattern,
    build_legal_pattern_catalog,
    oriented_endpoints,
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


class _BlockRobotGAT(nn.Module):
    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.layer1 = SparseGATLayer(NODE_DIM, hidden_dim, heads)
        self.layer2 = SparseGATLayer(hidden_dim, hidden_dim, heads)

    def encode_static(self, graph: BlockRobotGraph) -> torch.Tensor:
        hidden = torch.nn.functional.elu(
            self.layer1(graph.node_features, graph.edge_index, graph.edge_type,
                        incoming_edges=graph.incoming_edges)
        )
        return self.layer2(hidden, graph.edge_index, graph.edge_type,incoming_edges=graph.incoming_edges)


class BlockAssignmentGNN(_BlockRobotGAT):
    """Two-layer, hidden-64 GAT with a lightweight dynamic assignment head."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__(hidden_dim, heads)
        self.policy = nn.Sequential(
            nn.Linear(2 * hidden_dim + DYNAMIC_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

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

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=False, opportunity_features=False,parent_action_features=False):
        super().__init__(hidden_dim, heads)
        self.hidden_dim, self.heads = hidden_dim, heads
        self.opportunity_features=opportunity_features
        self.joint_feature_dim=JOINT_DYNAMIC_DIM+8*int(opportunity_features)
        self.parent_feature_dim=JOINT_PARENT_DIM+8*int(opportunity_features)
        self.parent_action_features=parent_action_features
        self.policy = nn.Identity()  # Assignment-only head is not used here.
        self.parent_head = nn.Sequential(nn.Linear(hidden_dim+self.parent_feature_dim,hidden_dim),
                                         nn.ReLU(),nn.Linear(hidden_dim,1))
        self.insertion_head = nn.Sequential(nn.Linear(4*hidden_dim+self.joint_feature_dim,hidden_dim),
                                            nn.ReLU(),nn.Linear(hidden_dim,1))
        if parent_action_features:
            self.parent_action_head=nn.Sequential(nn.Linear(9*hidden_dim+self.parent_feature_dim+2*self.joint_feature_dim+17,32),
                                                  nn.ReLU(),nn.Linear(32,1))
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

    def parent_action_scores(self,graph,embeddings,inputs):
        """One learned scalar per WHOLE or complete split action; no log-product."""
        rows=self.parent_action_rows(graph,embeddings,inputs)
        return self.parent_action_head(rows).squeeze(1)

    def parent_action_rows(self,graph,embeddings,inputs):
        """The same complete-action inputs, cacheable when the base is frozen."""
        if not self.parent_action_features:raise ValueError('complete parent scoring head is disabled')
        pooled={p:embeddings[graph.parent_block_indices[p]].mean(0) for p in dict.fromkeys(inputs['parent_ids'])}
        parent=torch.stack([pooled[p] for p in inputs['parent_ids']])
        indices=inputs['child_indices'];mask=inputs['child_mask']
        context=embeddings[indices.clamp_min(0)]*(indices>=0).unsqueeze(-1)
        context=context*mask[:,:,None,None]
        dynamic=self.scale_inputs(inputs['child_features'],'joint')*mask[:,:,None]
        features=torch.cat((parent,self.scale_inputs(inputs['parent_features'],'parent'),context.flatten(1),
            dynamic.flatten(1),inputs['after_features'],mask),dim=1)
        return features

    def contextualize_state(self, graph, embeddings, routes, directions):
        """Legacy policies retain their original static embeddings exactly."""
        return embeddings


class CoordinationJointInsertionGNN(JointInsertionGNN):
    """Refresh route/order/direction and same-rail context at every decision.

    These edges describe the student's current partial solution only. They do
    not assert schedulability: complete candidates still need B32 and the
    independent certifier. The original static policy/checkpoints are intact.
    """

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=True, opportunity_features=False,parent_action_features=False):
        super().__init__(hidden_dim, heads, normalize_inputs=normalize_inputs,opportunity_features=opportunity_features,
                         parent_action_features=parent_action_features)
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

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=True, opportunity_features=False,parent_action_features=False):
        super().__init__(hidden_dim, heads, normalize_inputs=normalize_inputs,opportunity_features=opportunity_features,
                         parent_action_features=parent_action_features)
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


def joint_opportunity_deltas(state,variants,blocks,patterns,config,*,insertion_limit=8,
                             check_initial_feasibility=True,reuse=True,normalize=True,stats=None):
    """Four action-level changes on the SAME post-action remaining block set.

    An insertion places one child. Its unplaced sibling stays in U_a and uses
    the real sibling-position restriction. These are block opportunities, not
    a claim that a selected first child guarantees completion of its parent.
    All membership queries use the existing insertion-existence function.

    Local reuse follows that function's dependencies: length, possible sibling
    neighbours, own head (including its alternate direction), and ALL other
    heads. Loads, internal non-head directions and travel do not affect it.
    reuse=False is the direct full calculation for small numerical comparisons.
    Nothing is cached across decision states or inference calls.
    """
    from .lns import has_joint_block_candidate
    from mrta_reference.geometry import oriented_endpoints
    before_placed={b for route in state.routes for b in route}
    cache={};queries=hits=0
    scale=max(1.,sum(config.process_time(b.length) for b in blocks.values())) if normalize else 1.
    def context(s):
        heads=tuple((r,oriented_endpoints(blocks[route[0]],s.directions[r][0])[0])
                    for r,route in enumerate(s.routes) if route)
        siblings=[]
        for route in s.routes:
            positions={}
            for i,b in enumerate(route):positions.setdefault(blocks[b].parent_id,[]).append((i,b))
            siblings.append(positions)
        return heads,siblings
    def available(s,b,r,ctx):
        nonlocal queries,hits
        if reuse:
            heads,siblings=ctx;route=s.routes[r]
            key=(b.block_id,r,len(route),route[0] if route else None,
                 s.directions[r][0] if route else None,tuple(p for p in heads if p[0]!=r),
                 tuple(i for i,sibling in siblings[r].get(b.parent_id,()) if sibling!=b.block_id))
            if key in cache:hits+=1;return cache[key]
        value=has_joint_block_candidate(s,b,patterns[b.parent_id],blocks,config,
            insertion_limit=insertion_limit,check_initial_feasibility=check_initial_feasibility,robot_id=r)
        queries+=1
        if reuse:cache[key]=value
        return value
    before_context=context(state) if reuse else None
    rows=[]
    for after,action in variants:
        placed={b for route in after.routes for b in route}
        if action.block_id in before_placed or placed!=before_placed|{action.block_id}:
            raise ValueError('opportunity delta expects one actual child insertion')
        remaining=[b for b in blocks.values() if b.block_id not in placed]
        after_context=context(after) if reuse else None
        delta=[]
        for r in range(4):
            before=sum(config.process_time(b.length) for b in remaining if available(state,b,r,before_context))
            post=sum(config.process_time(b.length) for b in remaining if available(after,b,r,after_context))
            delta.append((post-before)/scale)
        rows.append(tuple(delta))
    if stats is not None:
        stats['queries']=stats.get('queries',0)+queries
        stats['cache_hits']=stats.get('cache_hits',0)+hits
    return tuple(rows)


def joint_parent_inputs(state, pending, blocks, patterns, config, *, insertion_limit=8,check_initial_feasibility=True,
                        include_opportunities=False,filter_legal_parents=True):
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
        if filter_legal_parents and not has_joint_block_candidate(state,children[0],patterns[p],blocks,config,
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


def diverse_parent_indices(state,parent_ids,parent_logits,blocks,patterns,config,*,parent_limit=4,insertion_limit=8,
                           check_initial_feasibility=True):
    """Keep learned top-K and at most two legal qualification/load representatives."""
    from .lns import has_joint_block_candidate
    order=sorted(range(len(parent_ids)),key=lambda i:(-float(parent_logits[i]),i))
    chosen=order[:parent_limit];available=order[parent_limit:]
    information={}
    for i in order:
        children=sorted((b for b in blocks.values() if b.parent_id==parent_ids[i]),key=lambda b:b.block_id)
        robots=tuple(r for r in range(4) if has_joint_block_candidate(state,children[0],patterns[parent_ids[i]],blocks,config,
            insertion_limit=insertion_limit,robot_id=r,check_initial_feasibility=check_initial_feasibility))
        information[i]=(robots,sum(config.process_time(b.length) for b in children))
    if available and len(chosen)<8:
        scarce=max(available,key=lambda i:(information[i][1]/max(1,len(information[i][0])),information[i][1],
            -len(information[i][0]),float(parent_logits[i]),-i))
        chosen.append(scarce);available.remove(scarce)
    if available and len(chosen)<8:
        profiles={information[i][0] for i in chosen}
        low_load=max(available,key=lambda i:(information[i][0] not in profiles,
            max(state.loads)-min(state.loads[r] for r in information[i][0]),information[i][1],float(parent_logits[i]),-i))
        chosen.append(low_load)
    return tuple(chosen)


def diverse_child_indices(variants,logits,*,limit=2):
    """Keep neural best, then a different robot/load or insertion structure."""
    order=sorted(range(len(variants)),key=lambda i:(-float(logits[i]),i))
    if not order:return ()
    chosen=[order[0]]
    if limit==1:return tuple(chosen)
    first=variants[chosen[0]][1]
    other=[i for i in order[1:] if variants[i][1].robot!=first.robot]
    if other:chosen.append(min(other,key=lambda i:(max(variants[i][0].loads),-float(logits[i]),i)))
    else:
        other=[i for i in order[1:] if variants[i][1].position!=first.position]
        if other:chosen.append(other[0])
        elif len(order)>1:chosen.append(order[1])
    return tuple(chosen)


def parent_action_inputs(graph,state,plans,blocks,config):
    """Existing child features plus exact final partial loads/travel/head geometry."""
    from mrta_reference.geometry import oriented_endpoints
    n=len(plans);dim=plans[0]['child_features'].shape[1]
    indices=torch.full((n,2,4),-1,dtype=torch.long);features=torch.zeros(n,2,dim);mask=torch.zeros(n,2)
    scale=max(1.,sum(config.process_time(b.length) for b in blocks.values()));after_rows=[]
    for i,plan in enumerate(plans):
        count=len(plan['actions']);indices[i,:count]=plan['child_indices'];features[i,:count]=plan['child_features'];mask[i,:count]=1.
        after=plan['state'];moves=[sum(a) for a in after.arc_costs];pm=[p+m for p,m in zip(after.loads,moves)]
        overlaps=[];gaps=[]
        for left,right in ((0,1),(2,3)):
            if not after.routes[left] or not after.routes[right]:overlaps.append(0.);gaps.append(0.);continue
            upper=max(max(blocks[b].start[0],blocks[b].end[0]) for b in after.routes[left])
            lower=min(min(blocks[b].start[0],blocks[b].end[0]) for b in after.routes[right])
            overlaps.append(max(0.,upper+config.interference_dx-lower)/20.)
            starts=[oriented_endpoints(blocks[after.routes[r][0]],after.directions[r][0])[0][0] for r in (left,right)]
            gaps.append((starts[1]-starts[0])/20.)
        after_rows.append([*(p/scale for p in after.loads),*(m/scale for m in moves),max(after.loads)/scale,
                           max(pm)/scale,sum(moves)/scale,*overlaps,*gaps])
    return dict(parent_ids=tuple(p['parent_id'] for p in plans),parent_features=torch.stack([p['parent_features'] for p in plans]),
        child_indices=indices,child_features=features,child_mask=mask,after_features=torch.tensor(after_rows,dtype=torch.float32))


def joint_parent_action_candidates(graph,state,parent_ids,parent_logits,embeddings,model,blocks,patterns,config,
                                   *,parent_limit=4,child_limit=2,insertion_limit=8,current_embeddings=None,
                                   check_initial_feasibility=True,prefer_legal_start_recovery=True,timing=None,
                                   diverse=False,retain_inputs=False,parent_features=None):
    """Bounded complete parent actions scored by the existing conditional heads.

    Every softmax uses its full legal pool, before either top-k truncation.
    Branches retain immutable insertion states; only complete parents return.
    At most two children from the existing catalog, two branches per child.
    """
    from .lns import joint_block_candidates
    if not 1<=parent_limit<=8 or not 1<=child_limit<=2:
        raise ValueError('parent_limit must be 1..8 and child_limit must be 1..2')
    if not torch.isfinite(parent_logits).all():raise ValueError('NONFINITE_PARENT_LOGITS')
    parent_logp=torch.log_softmax(parent_logits,dim=0)
    ranking=sorted(range(len(parent_ids)),key=lambda i:(-float(parent_logits[i]),i))
    order=(diverse_parent_indices(state,parent_ids,parent_logits,blocks,patterns,config,
        parent_limit=parent_limit,insertion_limit=insertion_limit,check_initial_feasibility=
        check_initial_feasibility(state) if callable(check_initial_feasibility) else check_initial_feasibility)
        if diverse else ranking[:parent_limit])
    if retain_inputs and parent_features is None:
        ids,values=joint_parent_inputs(state,parent_ids,blocks,patterns,config,insertion_limit=insertion_limit,
            include_opportunities=getattr(model,'opportunity_features',False))
        parent_features=torch.stack([values[ids.index(p)] for p in parent_ids])
    plans=[]
    def check(s):
        return check_initial_feasibility(s) if callable(check_initial_feasibility) else check_initial_feasibility
    for index in order:
        rank=ranking.index(index)+1
        parent=parent_ids[index]
        children=sorted((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
        if len(children) not in (1,2):raise ValueError('existing parent catalog must contain one or two children')
        branches=[(state,(),float(parent_logp[index]),(),(),(),())]
        for child_index,child in enumerate(children):
            following=[]
            for partial,actions,score,logs,sizes,ranks,inputs in branches:
                tick=time.perf_counter() if timing is not None else 0.
                variants=joint_block_candidates(partial,child,patterns[parent],blocks,config,
                    insertion_limit=insertion_limit,check_initial_feasibility=check(partial),
                    prefer_legal_start_recovery=prefer_legal_start_recovery)
                if timing is not None:
                    timing['candidate_generation_s']=timing.get('candidate_generation_s',0.)+time.perf_counter()-tick
                    timing['insertion_candidates']=timing.get('insertion_candidates',0)+len(variants)
                if not variants:continue
                tick=time.perf_counter() if timing is not None else 0.
                indices,features=joint_candidate_inputs(graph,partial,variants,blocks,config,
                    include_opportunities=getattr(model,'opportunity_features',False),patterns=patterns,
                    insertion_limit=insertion_limit,check_initial_feasibility=check(partial))
                if timing is not None:
                    timing['feature_s']=timing.get('feature_s',0.)+time.perf_counter()-tick
                tick=time.perf_counter() if timing is not None else 0.
                context=(current_embeddings if child_index==0 and current_embeddings is not None else
                         model.contextualize_state(graph,embeddings,partial.routes,partial.directions))
                logits=model.joint_action_logits(context,indices,features)
                if timing is not None:
                    timing['network_s']=timing.get('network_s',0.)+time.perf_counter()-tick
                if not torch.isfinite(logits).all():raise ValueError('NONFINITE_INSERTION_LOGITS')
                logp=torch.log_softmax(logits,dim=0)
                selected=(diverse_child_indices(variants,logits,limit=child_limit) if diverse else
                          sorted(range(len(variants)),key=lambda i:(-float(logits[i]),i))[:child_limit])
                for child_rank,i in enumerate(selected,1):
                    after,action=variants[i];value=float(logp[i])
                    retained=inputs+((indices[i].clone(),features[i].clone()),) if retain_inputs else ()
                    following.append((after,actions+(action,),score+value,logs+(value,),sizes+(len(variants),),ranks+(child_rank,),retained))
            branches=following
        for after,actions,score,logs,sizes,ranks,inputs in branches:
            plan=dict(parent_id=parent,parent_rank=rank,state=after,actions=actions,score=score,
                parent_log_probability=float(parent_logp[index]),child_log_probabilities=logs,
                child_pool_sizes=sizes,child_ranks=ranks)
            if retain_inputs:plan.update(parent_features=parent_features[index].clone(),
                child_indices=torch.stack([x[0] for x in inputs]),child_features=torch.stack([x[1] for x in inputs]))
            plans.append(plan)
    return tuple(plans)


def _decode_joint(reference, routes, directions, pending_parent_ids, model, config,
                  *, insertion_limit, choose,allow_partial_start_conflicts=False,prefer_legal_start_recovery=True,
                  parent_selection_mode='learned',action_selection_mode='gnn',action_observer=None,
                  forced_action=None,parent_observer=None,joint_greedy=False,parent_joint=False,
                  parent_candidate_limit=4,parent_action_observer=None,parent_cmax=False,diverse_candidates=False):
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
                if parent_joint:
                    plans=joint_parent_action_candidates(graph,state,ids,parent_logits,embeddings,model,blocks,patterns,config,
                        parent_limit=parent_candidate_limit,child_limit=2,insertion_limit=insertion_limit,
                        current_embeddings=current_embeddings,check_initial_feasibility=check_start,
                        prefer_legal_start_recovery=prefer_legal_start_recovery,diverse=diverse_candidates,
                        retain_inputs=parent_cmax,parent_features=features)
                    if not plans:raise _JointActionUnavailable('NO_LEGAL_PARENT_COMPLETION')
                    if parent_cmax:
                        inputs=parent_action_inputs(graph,state,plans,blocks,config)
                        scores=model.parent_action_scores(graph,current_embeddings,inputs)
                        if not torch.isfinite(scores).all():raise ValueError('NONFINITE_PARENT_ACTION_SCORES')
                        best=plans[choose(scores,'parent_action')]
                    else:best=max(plans,key=lambda p:p['score'])
                    if parent_action_observer is not None:
                        parent_action_observer(state,plans,best)
                    # One commit after both children have been evaluated. A
                    # failed speculative branch never appends a real action.
                    state=best['state'];actions.extend(best['actions'])
                    pending.remove(best['parent_id'])
                    continue
                elif joint_greedy:
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
                                 forced_action=None,parent_observer=None,parent_candidate_limit=4,parent_action_observer=None,
                                 diverse_candidates=False):
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
    'parent_joint' evaluates complete parents within top-K (K<=8), with top-two
    child branches and the conditional probability of every split child.
    'parent_cmax' uses the learned complete-action scalar and bounded diverse
    proposals. 'parent_cmax_sample' samples these same scores at temperature 1.
    """
    if decode_mode not in ('greedy','sample','joint_greedy','parent_joint','parent_cmax','parent_cmax_sample'):
        raise ValueError('unknown GNN decode_mode')
    if decode_mode in ('joint_greedy','parent_joint','parent_cmax','parent_cmax_sample') and (parent_selection_mode!='learned' or action_selection_mode!='gnn'):
        raise ValueError('joint decoding requires the learned parent and GNN insertion heads')
    if decode_mode.startswith('parent_cmax') and not getattr(model,'parent_action_features',False):
        raise ValueError('parent_cmax requires the complete action scoring head')
    generator=None
    if decode_mode in ('sample','parent_cmax_sample'):
        generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    def choose(logits,kind):
        if decode_mode in ('greedy','joint_greedy','parent_joint','parent_cmax'):
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
                             joint_greedy=decode_mode=='joint_greedy',parent_joint=decode_mode in ('parent_joint','parent_cmax','parent_cmax_sample'),
                             parent_candidate_limit=parent_candidate_limit,parent_action_observer=parent_action_observer,
                             parent_cmax=decode_mode.startswith('parent_cmax'),
                             diverse_candidates=diverse_candidates or decode_mode.startswith('parent_cmax'))


class CompleteParentGNN(ModeJointInsertionGNN):
    """Two existing GAT layers and one joint mode/assignment/insertion scorer."""

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=True,
                 opportunity_features=True, parent_action_features=True, remaining_workload_features=False):
        super().__init__(hidden_dim, heads, normalize_inputs=normalize_inputs,
                         opportunity_features=opportunity_features, parent_action_features=True)
        width=self.parent_action_head[0].in_features
        self.remaining_workload_features=remaining_workload_features
        self.compact_parent_limit=6 if remaining_workload_features else None
        self.parent_action_head=nn.Sequential(nn.Linear(width+12+4*int(remaining_workload_features),32),nn.ReLU(),nn.Linear(32,1))

    def parent_action_rows(self,graph,embeddings,inputs):
        rows=torch.cat((super().parent_action_rows(graph,embeddings,inputs),inputs['mode_features']),1)
        if self.remaining_workload_features:
            rows=torch.cat((rows,inputs['remaining_eligible_workload']),1)
        return rows


def complete_parent_space(parents, config, *, compact=False):
    """Encode every legal pattern once; aliases distinguish alternative geometries.

    Routes/actions retain native block IDs. Only GAT node IDs are namespaced,
    so mutually exclusive modes never overwrite one another's geometry.
    """
    parents=tuple(sorted(parents,key=lambda p:p.parent_id))
    catalog=build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)
    nodes=[];indices={};options={}
    for parent in parents:
        for k,pattern in enumerate(catalog[parent.parent_id]):
            children=blocks_for_pattern(parent,pattern,config)
            options[pattern]=children
            for child in children:
                alias=f'{parent.parent_id}@{k}/{child.block_id}'
                indices[pattern,child.block_id]=len(nodes)
                nodes.append(replace(child,block_id=alias))
    graph=build_block_robot_graph(nodes,tuple(p for ps in catalog.values() for p in ps),config,
                                  include_parent_edges=True)
    space=dict(parents=parents,catalog=catalog,options=options,indices=indices,graph=graph)
    if compact:
        flexibility={};workload={}
        for parent in parents:
            classes={};flex=0
            for pattern in catalog[parent.parent_id]:
                row=[0.]*4;count=1
                for child in options[pattern]:
                    robots=((0 if pattern.rail is Rail.UPPER else 2)+int(child.block_id.endswith('::1')), ) if pattern.kind is SplitKind.X_SPLIT else _eligible_robots(child,config)
                    count*=len(robots)
                    work=config.process_time(child.length)/len(robots)
                    for robot in robots:row[robot]+=work
                flex+=count;classes.setdefault(pattern.kind,[]).append(row)
            # Equal weight per mode KIND, then equal weight per cut in that kind.
            workload[parent.parent_id]=tuple(sum(sum(row[r] for row in rows)/len(rows)
                for rows in classes.values())/len(classes) for r in range(4))
            flexibility[parent.parent_id]=flex
        space.update(flexibility=flexibility,remaining_parent_workload=workload,
            original_processing={p.parent_id:config.process_time(p.length) for p in parents})
    return space


class ParentAssignment(NamedTuple):
    """A legal mode and its owners in blocks_for_pattern order (spatial for X)."""

    parent_id: str
    pattern: SplitPattern
    robots: tuple[int, ...]


def _blocks_by_parent(blocks):
    grouped = {}
    for block in blocks.values():
        grouped.setdefault(block.parent_id, []).append(block)
    return grouped


def parent_assignment_targets(solution: CanonicalSolution, config: ScientificConfig) -> tuple[ParentAssignment, ...]:
    """Read ownership from a complete solution without preserving its route order."""
    children = _blocks_by_parent(block_map(solution, config, scope=FORMAL_SCOPE_V2))
    owners = assignment_targets(solution)
    return tuple(ParentAssignment(p.parent_id, p, tuple(owners[b.block_id] for b in children[p.parent_id]))
                 for p in solution.patterns)


def decode_parent_assignments(
    parents: Sequence[ParentWeld], assignments: Sequence[ParentAssignment], config: ScientificConfig,
    *, coordination_aware: bool = False, decision_observer=None,
) -> JointCompletionResult:
    """Insert fixed assignments in LPT order with optional nominal coordination scoring."""
    from .lns import NominalRouteCoordination, joint_block_candidates, joint_block_robots, oriented_route_state

    started = time.perf_counter()
    pending = {parent.parent_id for parent in parents}
    assigned = {action.parent_id: action for action in assignments}
    if len(assigned) != len(assignments) or assigned.keys() != pending:
        raise ValueError('exactly one assignment per parent is required')
    if any(action.parent_id != action.pattern.parent_id for action in assignments):
        raise ValueError('assignment and pattern parent differ')
    patterns = tuple(assigned[p].pattern for p in sorted(assigned))
    blocks = block_map(JointGeometry(tuple(parents), patterns), config, scope=FORMAL_SCOPE_V2)
    children = _blocks_by_parent(blocks)
    priorities = {}
    inward_x = [-math.inf] * 4
    side = (1, -1, 1, -1)
    for parent_id, pattern, robots in assignments:
        siblings = children[parent_id]
        choices = [joint_block_robots(block, pattern, config) for block in siblings]
        if len(robots) != len(siblings):
            raise ValueError(f'{parent_id}: one owner per selected block is required')
        for block, robot, eligible in zip(siblings, robots, choices):
            if robot not in eligible or not robot_is_eligible(block, robot, config):
                raise ValueError(f'illegal assignment {block.block_id} -> R{robot}')
            inward_x[robot] = max(inward_x[robot], side[robot] * block.start[0], side[robot] * block.end[0])
        priorities[parent_id] = (-sum(config.process_time(b.length) for b in siblings),
                                 math.prod(map(len, choices)), parent_id)

    empty = ((),) * 4
    state = oriented_route_state(empty, empty, blocks, config)
    nominal = NominalRouteCoordination.from_state(state, blocks, config) if coordination_aware else None
    actions, deferred = [], []

    def insert(block, pattern, robot):
        nonlocal state, nominal
        # Partial heads may conflict before a later block supplies a new head.
        # The common generator prefers exact legal starts whenever available.
        variants = joint_block_candidates(state, block, pattern, blocks, config, robot_id=robot,
                                          check_initial_feasibility=False)
        if not variants:
            raise ValueError(f'{block.parent_id}: same-robot siblings require a separating block')

        def travel_score(row):
            candidate, action = row
            route, bits = candidate.routes[robot], candidate.directions[robot]
            first, _ = oriented_endpoints(blocks[route[0]], bits[0])
            travel = sum(candidate.arc_costs[robot])
            # Prefer room at the start for a partner to traverse its range.
            # This distance term ranks candidates without rejecting overlaps.
            shortfall = max(0., side[robot] * first[0] + inward_x[robot ^ 1] + config.interference_dx)
            return travel + shortfall / config.empty_speed, travel, action.position, bits

        if nominal is None:
            state, action = min(variants, key=travel_score)
        else:
            def coordinated_candidate(row):
                updated = nominal.with_changed_route(row[0], robot, blocks, config)
                features = updated.features()
                score = (features['exposure_score'], max(features['same_rail_order_seconds']), *travel_score(row))
                return score, row, updated

            _, (state, action), nominal = min(map(coordinated_candidate, variants), key=lambda candidate: candidate[0])
        actions.append(action)

    for parent_id in sorted(pending, key=priorities.__getitem__):
        if decision_observer is not None:
            decision_observer(state, blocks, frozenset(pending))
        _, pattern, robots = assigned[parent_id]
        for block, robot in zip(children[parent_id], robots):
            route = state.routes[robot]
            if len(route) == 1 and blocks[route[0]].parent_id == parent_id:
                # Adjacent siblings would be merged by canonicalization.
                deferred.append((block, pattern, robot))
            else:
                insert(block, pattern, robot)
        pending.remove(parent_id)
    for block, pattern, robot in deferred:
        insert(block, pattern, robot)

    solution = canonicalize(parents, patterns, dict(enumerate(state.routes)), config, scope=FORMAL_SCOPE_V2)
    return JointCompletionResult(solution, state.directions, tuple(actions), time.perf_counter() - started,
                                 oriented_state=state)


@dataclass(frozen=True)
class ParentAssignmentSpace:
    graph: BlockRobotGraph
    parent_ids: tuple[str, ...]
    actions: tuple[ParentAssignment, ...]
    action_parents: torch.Tensor
    child_nodes: torch.Tensor
    owners: torch.Tensor
    active_children: torch.Tensor
    modes: torch.Tensor
    processing: torch.Tensor
    children: Mapping[SplitPattern, tuple[WeldingBlock, ...]]
    block_nodes: Mapping[tuple[SplitPattern, str], int]
    remaining_processing: torch.Tensor

    @cached_property
    def action_index(self):
        return {action: index for index, action in enumerate(self.actions)}


def parent_assignment_space(parents: Sequence[ParentWeld], config: ScientificConfig) -> ParentAssignmentSpace:
    """Prepare compact rows for all legal mode/owner tuples once per geometry."""
    from .lns import joint_block_robots

    space = complete_parent_space(parents, config, compact=True)
    parent_ids = tuple(p.parent_id for p in space['parents'])
    actions, action_parents, child_nodes, owners, masks, modes, processing = [], [], [], [], [], [], []
    for index, parent_id in enumerate(parent_ids):
        for pattern in space['catalog'][parent_id]:
            children = space['options'][pattern]
            nodes = tuple(space['indices'][pattern, b.block_id] for b in children)
            times = tuple(config.process_time(b.length) / 2500. for b in children)
            for robots in product(*(joint_block_robots(b, pattern, config) for b in children)):
                actions.append(ParentAssignment(parent_id, pattern, robots))
                action_parents.append(index)
                child_nodes.append(nodes if len(nodes) == 2 else (*nodes, 0))
                owners.append(robots if len(robots) == 2 else (*robots, 0))
                masks.append((True, len(robots) == 2))
                modes.append([float(pattern.kind is kind) for kind in SplitKind]
                             + [float(pattern.mandatory), pattern.t if pattern.t is not None else .5])
                loads = [0.] * 4
                for robot, seconds in zip(robots, times):
                    loads[robot] += seconds
                processing.append(loads)
    return ParentAssignmentSpace(space['graph'], parent_ids, tuple(actions),
                                 torch.tensor(action_parents, dtype=torch.long),
                                 torch.tensor(child_nodes, dtype=torch.long).reshape(-1, 2),
                                 torch.tensor(owners, dtype=torch.long).reshape(-1, 2),
                                 torch.tensor(masks, dtype=torch.bool).reshape(-1, 2),
                                 torch.tensor(modes, dtype=torch.float32).reshape(-1, 5),
                                 torch.tensor(processing, dtype=torch.float32).reshape(-1, 4),
                                 space['options'], space['indices'],
                                 torch.tensor([space['remaining_parent_workload'][p] for p in parent_ids],
                                              dtype=torch.float32).reshape(-1, 4) / 2500.)


class ParentAssignmentGNN(_BlockRobotGAT):
    """Shared two-layer GAT and a single score for each legal joint action."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__(hidden_dim, heads)
        # Each child sees its own and its robot's embeddings, plus seven state
        # features from its robot and that robot's same-rail partner.
        self.policy = nn.Sequential(nn.Linear(4 * hidden_dim + 4 * 7 + 5 + 4, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def assignment_logits(self, space, embeddings, state, blocks, pending, config):
        """Return indices into space.actions and batched scores for pending parents."""
        indices, inputs = self.assignment_rows(space, embeddings, state, blocks, pending, config)
        return indices, self.policy(inputs).squeeze(1)

    def assignment_rows(self, space, embeddings, state, blocks, pending, config):
        device = embeddings.device
        pending_mask = torch.tensor([p in pending for p in space.parent_ids], device=device, dtype=torch.bool)
        indices = torch.where(pending_mask[space.action_parents.to(device)])[0]
        rows = []
        for robot, (route, bits) in enumerate(zip(state.routes, state.directions)):
            if route:
                head, _ = oriented_endpoints(blocks[route[0]], bits[0])
                _, tail = oriented_endpoints(blocks[route[-1]], bits[-1])
            else:
                head = tail = _home(robot, config)
            rows.append((state.loads[robot] / 2500., sum(state.arc_costs[robot]) / 2500.,
                         head[0] / 20., head[1] / 12., tail[0] / 20., tail[1] / 12., float(bool(route))))
        robot_state = embeddings.new_tensor(rows)
        owners = space.owners.to(device)[indices]
        children = torch.cat((embeddings[space.child_nodes.to(device)[indices]],
                              embeddings[space.graph.robot_offset + owners],
                              robot_state[owners], robot_state[owners ^ 1]), dim=-1)
        children = (children * space.active_children.to(device)[indices, :, None]).flatten(1)
        projected = space.processing.to(embeddings)[indices] + robot_state[:, 0]
        inputs = torch.cat((children, space.modes.to(embeddings)[indices], projected), dim=1)
        return indices, inputs


COARSE_ASSIGNMENT_LIMIT = 8
FINE_INSERTION_LIMIT = 6
COORDINATION_FEATURES = 9


class CoarseToFineGNN(ParentAssignmentGNN):
    """One geometry encoder, an assignment logit and a complete-insertion logit."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4):
        super().__init__(hidden_dim, heads)
        self.geometry_skip = nn.Linear(NODE_DIM, hidden_dim)
        width = self.policy[0].in_features + 9
        self.policy = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.insertion_head = nn.Sequential(nn.Linear(width + 4 * hidden_dim + 22 + COORDINATION_FEATURES, hidden_dim),
                                            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode_static(self, graph):
        return super().encode_static(graph) + self.geometry_skip(graph.node_features)

    def assignment_rows(self, space, embeddings, state, blocks, pending, config):
        indices, rows = super().assignment_rows(space, embeddings, state, blocks, pending, config)
        owners = space.owners.to(indices.device)[indices]
        active = space.active_children.to(indices.device)[indices]
        occupied = torch.tensor([bool(route) for route in state.routes], device=indices.device)
        # Same-robot siblings need a pre-existing block between them to be
        # representable as a committed split in the canonical solution.
        keep = ~(active[:, 1] & (owners[:, 0] == owners[:, 1]) & ~occupied[owners[:, 0]])
        indices, rows = indices[keep], rows[keep]
        pending_mask = torch.tensor([p in pending for p in space.parent_ids], device=indices.device)
        work = space.remaining_processing.to(rows)
        remaining = (work[pending_mask].sum(0) - work[space.action_parents.to(indices.device)[indices]]).clamp_min(0.)
        expected = rows[:, -4:] + remaining
        shares = expected / expected.sum(1, keepdim=True).clamp_min(1e-6)
        return indices, torch.cat((rows, remaining, shares, shares.amax(1, keepdim=True)), dim=1)

    def insertion_logits(self, rows, *, use_coordination=True):
        if not use_coordination:
            rows = torch.cat((rows[:, :-COORDINATION_FEATURES],
                              torch.zeros_like(rows[:, -COORDINATION_FEATURES:])), dim=1)
        return self.insertion_head(rows).squeeze(1)

    def complete_logits(self, coarse_rows, fine_rows, *, use_coordination=True):
        return self.policy(coarse_rows).squeeze(1) + self.insertion_logits(fine_rows, use_coordination=use_coordination)


def shortlist_assignments(space, indices, logits, *, owner_alternatives=False):
    """Retain learned scores, with slots for distinct parents and mode families."""
    order = indices[torch.argsort(logits, descending=True, stable=True)].tolist()
    chosen, parents, kinds = [], set(), set()
    for index in order:
        action = space.actions[index]
        if action.parent_id in parents:
            continue
        chosen.append(index)
        parents.add(action.parent_id)
        kinds.add(action.pattern.kind)
        if len(chosen) == COARSE_ASSIGNMENT_LIMIT // 2:
            break
    for index in order:
        kind = space.actions[index].pattern.kind
        if kind not in kinds:
            chosen.append(index)
            kinds.add(kind)
    if owner_alternatives:
        for primary in chosen[:2]:
            pattern = space.actions[primary].pattern
            alternate = next((i for i in order if space.actions[i].pattern == pattern and i not in chosen), None)
            if alternate is not None:
                chosen.append(alternate)
    return tuple(dict.fromkeys((*chosen, *order)))[:COARSE_ASSIGNMENT_LIMIT]


class AssignmentInsertion(NamedTuple):
    assignment_index: int
    state: object
    actions: tuple


def assignment_insertions(space, index, state, blocks, config):
    """Expand just one fixed-owner assignment and return only atomic parent actions."""
    from .lns import joint_block_candidates

    assignment = space.actions[index]
    children = space.children[assignment.pattern]
    local_blocks = dict(blocks, **{b.block_id: b for b in children})
    branches = [(state, ())]
    for child, robot in zip(children, assignment.robots):
        limit = FINE_INSERTION_LIMIT if len(children) == 1 else (3 if child is children[0] else 2)
        next_branches = []
        for partial, actions in branches:
            variants = joint_block_candidates(partial, child, assignment.pattern, local_blocks, config,
                                              robot_id=robot, check_initial_feasibility=False)
            for i in structural_child_indices(variants, limit=limit):
                after, action = variants[i]
                next_branches.append((after, (*actions, action)))
        branches = next_branches
    return tuple(AssignmentInsertion(index, after, actions) for after, actions in branches)


def assignment_insertion_rows(space, embeddings, state, blocks, selected, nominal, plans, coarse_rows, config):
    """Use committed and proposed geometry for local neighbors and complete outcomes."""
    prior_nodes = {b.block_id: space.block_nodes[pattern, b.block_id]
                   for pattern in selected.values() for b in space.children[pattern]}
    scales = max(1., float(space.remaining_processing.sum()) * 2500. / 4.)
    neighbor_indices, scalars, contexts = [], [], []
    for plan in plans:
        assignment = space.actions[plan.assignment_index]
        new_blocks = {b.block_id: b for b in space.children[assignment.pattern]}
        new_nodes = {b: space.block_nodes[assignment.pattern, b] for b in new_blocks}
        after = plan.state
        indices, local = [], []
        for action in plan.actions:
            route, bits = after.routes[action.robot], after.directions[action.robot]
            k = route.index(action.block_id)
            for j in (k - 1, k + 1):
                if 0 <= j < len(route):
                    neighbor = route[j]
                    indices.append(new_nodes[neighbor] if neighbor in new_nodes else prior_nodes[neighbor])
                else:
                    indices.append(-1)
            local.extend((k / max(1, len(route) - 1), float(bits[k]), bits[k-1] if k else 0.,
                          bits[k+1] if k + 1 < len(route) else 0., len(action.neighbour_changes) / 2.))
        if len(plan.actions) == 1:
            indices.extend((-1, -1))
            local.extend((0.,) * 5)
        moves = tuple(sum(arcs) for arcs in after.arc_costs)
        p_move = tuple(p + m for p, m in zip(after.loads, moves))
        outcome = (*after.loads, max(after.loads), *p_move, max(p_move), sum(moves), sum(moves) - state.travel)
        if nominal is None:
            updated, coordination = None, (0.,) * COORDINATION_FEATURES
        else:
            robots = tuple(dict.fromkeys(assignment.robots))
            updated = nominal.with_changed_routes(after, robots, dict(blocks, **new_blocks), config)
            feature = updated.features()
            coordination = (*feature['same_rail_order_seconds'], *feature['proximity_seconds'], feature['exposure_score'])
        neighbor_indices.append(indices)
        scalars.append((*local, *(v / scales for v in outcome), *(v / scales for v in coordination)))
        contexts.append(updated)
    neighbors = torch.tensor(neighbor_indices, dtype=torch.long, device=embeddings.device)
    context_rows = embeddings[neighbors.clamp_min(0)] * (neighbors >= 0).unsqueeze(-1)
    return torch.cat((coarse_rows, context_rows.flatten(1), embeddings.new_tensor(scalars)), dim=1), tuple(contexts)


def legal_assignment_starts(plans, blocks, space, config):
    """Prefer an exact legal start across complete parent actions, as in fixed decoding."""
    from .direction import _initial_error

    indices = []
    for i, plan in enumerate(plans):
        pattern = space.actions[plan.assignment_index].pattern
        proposed = {b.block_id: b for b in space.children[pattern]}
        points = {r: oriented_endpoints(proposed.get(route[0]) or blocks[route[0]], bits[0])[0]
                  for r, (route, bits) in enumerate(zip(plan.state.routes, plan.state.directions)) if route}
        if _initial_error(points, config) is None:
            indices.append(i)
    return tuple(indices) or tuple(range(len(plans)))


@torch.no_grad()
def construct_coarse_to_fine_gnn(parents, model, config, *, space=None, embeddings=None,
                                state=None, selected=None, use_coordination=True, owner_alternatives=False, observer=None):
    """Construct one solution, or resume a frozen rollout after a forced parent action."""
    from .lns import NominalRouteCoordination, oriented_route_state

    started = time.perf_counter()
    space = parent_assignment_space(parents, config) if space is None else space
    embeddings = model.encode_static(space.graph) if embeddings is None else embeddings
    selected = {} if selected is None else dict(selected)
    blocks = {b.block_id: b for pattern in selected.values() for b in space.children[pattern]}
    empty = ((),) * 4
    state = oriented_route_state(empty, empty, blocks, config) if state is None else state
    nominal = NominalRouteCoordination.from_state(state, blocks, config) if use_coordination else None
    pending = set(space.parent_ids) - selected.keys()
    actions = []
    while pending:
        indices, coarse_rows = model.assignment_rows(space, embeddings, state, blocks, pending, config)
        coarse_logits = model.policy(coarse_rows).squeeze(1)
        order = shortlist_assignments(space, indices, coarse_logits, owner_alternatives=owner_alternatives)
        plans = tuple(plan for index in order for plan in assignment_insertions(space, index, state, blocks, config))
        if not plans:
            return JointCompletionResult(None, None, tuple(actions), time.perf_counter() - started,
                                         rejection_reason='NO_COMPLETE_PARENT_ACTION')
        plans = tuple(plans[i] for i in legal_assignment_starts(plans, blocks, space, config))
        by_index = {index: i for i, index in enumerate(indices.tolist())}
        plan_indices = [by_index[plan.assignment_index] for plan in plans]
        fine_rows, contexts = assignment_insertion_rows(space, embeddings, state, blocks, selected, nominal, plans,
                                                      coarse_rows[plan_indices], config)
        logits = coarse_logits[plan_indices] + model.insertion_logits(fine_rows, use_coordination=use_coordination)
        best = int(logits.argmax())
        if observer is not None:
            observer(state, blocks, selected, pending, nominal, indices, coarse_rows, order, plans, fine_rows, logits, best)
        chosen = plans[best]
        assignment = space.actions[chosen.assignment_index]
        state, nominal = chosen.state, contexts[best]
        selected[assignment.parent_id] = assignment.pattern
        blocks.update((b.block_id, b) for b in space.children[assignment.pattern])
        pending.remove(assignment.parent_id)
        actions.extend(chosen.actions)
    solution = canonicalize(parents, tuple(selected.values()), dict(enumerate(state.routes)), config, scope=FORMAL_SCOPE_V2)
    return JointCompletionResult(solution, state.directions, tuple(actions), time.perf_counter() - started,
                                 oriented_state=state)


def gnn_parent_orders(space,state,pending,selected,model,config,*,proposal_graph_mode='native'):
    """Reuse the existing parent head on inference-available task graphs.

    Retained parents use their committed modes; uncommitted parents use the
    catalog default, never a future teacher mode. The full action pool still
    considers every legal mode. One set of GAT weights serves both graphs.
    """
    _,blocks,patterns=complete_parent_view(space,selected)
    key=tuple(patterns.values())
    cached=space.get('parent_proposal_graph')
    if cached is None or cached[0]!=key:
        graph=build_block_robot_graph(tuple(sorted(blocks.values(),key=lambda b:b.block_id)),key,config,
                                     include_parent_edges=True)
        space['parent_proposal_graph']=(key,graph)
    else:graph=cached[1]
    ids,features=joint_parent_inputs(state,pending,blocks,patterns,config,
        include_opportunities=getattr(model,'opportunity_features',False),filter_legal_parents=False)
    if proposal_graph_mode=='native':
        embeddings=model.contextualize_state(graph,model.encode_static(graph),state.routes,state.directions)
        logits=model.parent_logits(graph,embeddings,ids,features)
    elif proposal_graph_mode in ('catalog_mean','catalog_max'):
        view,_,_=complete_parent_view(space,selected)
        embeddings=model.contextualize_state(view,model.encode_static(space['graph']),state.routes,state.directions)
        if proposal_graph_mode=='catalog_mean':
            logits=model.parent_logits(space['graph'],embeddings,ids,features)
        else:
            contexts=[];dynamic=[];owners=[]
            total=max(1.,sum(config.process_time(b.length) for b in blocks.values()))
            moves=[sum(arcs) for arcs in state.arc_costs]
            for index,parent in enumerate(ids):
                for pattern in space['catalog'][parent]:
                    children=space['options'][pattern]
                    context=embeddings[[space['indices'][pattern,b.block_id] for b in children]].mean(0)
                    row=features[index].clone()
                    robots=((0 if pattern.rail is Rail.UPPER else 2),) if pattern.kind is SplitKind.X_SPLIT else _eligible_robots(children[0],config)
                    row[8]=sum(config.process_time(b.length) for b in children)/total
                    row[9]=min(state.loads[r]+moves[r] for r in robots)/total
                    row[11]=len(children)/2
                    contexts.append(context);dynamic.append(row);owners.append(index)
            values=model.parent_head(torch.cat((torch.stack(contexts),model.scale_inputs(torch.stack(dynamic),'parent')),1)).squeeze(1)
            logits=torch.stack([values[[i for i,p in enumerate(owners) if p==index]].max() for index in range(len(ids))])
    else:raise ValueError('unknown proposal graph representation')
    if not torch.isfinite(logits).all():raise ValueError('NONFINITE_PARENT_PROPOSAL_LOGITS')
    gnn=tuple(ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(logits[i]),ids[i])))
    mrv=tuple(sorted(pending,key=lambda p:(space['flexibility'][p],-space['original_processing'][p],p)))
    lpt=tuple(sorted(pending,key=lambda p:(-space['original_processing'][p],space['flexibility'][p],p)))
    return dict(gnn=gnn,mrv=mrv,lpt=lpt)


def compact_parent_ids(space,pending,*,parent_limit=4,strategy=None,orders=None):
    """Legacy MRV/LPT selection or a measured existing-head ranking."""
    if strategy is not None:
        if parent_limit not in (4,6,8,12,16):raise ValueError('ranked parent limit must be 4,6,8,12 or 16')
        if strategy not in ('gnn','mrv','lpt','gnn_mrv_lpt'):raise ValueError('unknown parent proposal strategy')
        if strategy!='gnn_mrv_lpt':return tuple(orders[strategy][:parent_limit])
        chosen=list(orders['gnn'][:max(0,parent_limit-2)])
        for name in ('mrv','lpt'):
            extra=next((p for p in orders[name] if p not in chosen),None)
            if extra is not None:chosen.append(extra)
        for p in orders['gnn']:
            if len(chosen)>=min(parent_limit,len(pending)):break
            if p not in chosen:chosen.append(p)
        return tuple(chosen[:parent_limit])
    if parent_limit not in (4,6):raise ValueError('compact parent limit must be 4 or 6')
    count=parent_limit//2
    scarce=sorted(pending,key=lambda p:(space['flexibility'][p],-space['original_processing'][p],p))[:count]
    long=sorted(pending,key=lambda p:(-space['original_processing'][p],space['flexibility'][p],p))[:count]
    return tuple(dict.fromkeys(scarce+long))


def remaining_eligible_workload(space,pending):
    """Category-averaged remaining eligibility shares, not a load lower bound."""
    pending=tuple(sorted(pending))
    work=[sum(space['remaining_parent_workload'][p][r] for p in pending) for r in range(4)]
    total=sum(work)
    return torch.tensor([w/total if total else 0. for w in work],dtype=torch.float32)


def complete_parent_view(space, selected):
    patterns={p.parent_id:selected.get(p.parent_id,space['catalog'][p.parent_id][0]) for p in space['parents']}
    blocks={b.block_id:b for p in patterns.values() for b in space['options'][p]}
    index={b.block_id:space['indices'][p,b.block_id] for p in patterns.values() for b in space['options'][p]}
    return replace(space['graph'],block_index=index),blocks,patterns


def structural_child_indices(variants, *, limit):
    """Bounded weight-independent representatives of robot, direction and position.

    Preserve the cheapest exact neighbour configuration for each position and
    direction, then round-robin robots/directions before filling by travel.
    No teacher action or old policy logit participates in this selection.
    """
    order=sorted(range(len(variants)),key=lambda i:(variants[i][0].travel,i))
    groups={}
    for i in order:
        a=variants[i][1]
        groups.setdefault((a.robot,a.direction),[]).append(i)
    chosen=[]
    for depth in range(limit):
        for key in sorted(groups):
            values=groups[key]
            # Distinct insertion positions before further neighbour variants.
            positions={};rest=[]
            for i in values:
                position=variants[i][1].position
                if position not in positions:positions[position]=i
                else:rest.append(i)
            spread=list(positions.values())+rest
            if depth<len(spread):chosen.append(spread[depth])
            if len(chosen)>=limit:return tuple(chosen)
    return tuple(chosen)


def complete_parent_candidates(space,state,pending,selected,config,*,first_limit=16,second_limit=8,
                               parent_limit=None,include_remaining_workload=False,candidate_parent_ids=None,
                               remaining_after_parent=False):
    """Every unassigned parent/mode gets complete bounded sequential proposals.

    WHOLE exposes its bounded insertion pool. Split branches retain at most
    first_limit*second_limit actions, never the full child Cartesian product.
    """
    from .lns import joint_block_candidates
    plans=[]
    parents=(tuple(candidate_parent_ids) if candidate_parent_ids is not None else sorted(pending)
             if parent_limit is None else compact_parent_ids(space,pending,parent_limit=parent_limit))
    if not set(parents)<=set(pending):raise ValueError('candidate parent is not pending')
    remaining=remaining_eligible_workload(space,pending) if include_remaining_workload and not remaining_after_parent else None
    for parent in parents:
        if include_remaining_workload and remaining_after_parent:
            remaining=remaining_eligible_workload(space,set(pending)-{parent})
        for pattern in space['catalog'][parent]:
            graph,blocks,patterns=complete_parent_view(space,dict(selected,**{parent:pattern}))
            children=space['options'][pattern]
            scale=max(1.,sum(config.process_time(b.length) for b in blocks.values()))
            moves=[sum(a) for a in state.arc_costs]
            legal=_eligible_robots(children[0],config)
            pf=torch.tensor([*(p/scale for p in state.loads),*(v/scale for v in moves),
                sum(config.process_time(b.length) for b in children)/scale,
                min(state.loads[r]+moves[r] for r in legal)/scale,len(pending)/len(patterns),len(children)/2,
                *([0.]*8)],dtype=torch.float32)
            branches=[(state,(),(),())]
            for child_no,child in enumerate(children):
                following=[]
                for partial,actions,indices,features in branches:
                    variants=joint_block_candidates(partial,child,pattern,blocks,config,insertion_limit=8)
                    keep=(range(len(variants)) if len(children)==1 and parent_limit is None else
                          structural_child_indices(variants,limit=first_limit if child_no==0 else second_limit))
                    kept=tuple(variants[i] for i in keep)
                    if not kept:continue
                    ix,fs=joint_candidate_inputs(graph,partial,kept,blocks,config)
                    fs=torch.cat((fs,torch.zeros(len(fs),8)),1)
                    for i,(after,action) in enumerate(kept):
                        following.append((after,actions+(action,),indices+(ix[i],),features+(fs[i],)))
                branches=following
            lengths=[b.length for b in children];processing=[config.process_time(v) for v in lengths]
            mode=torch.tensor([float(pattern.kind is SplitKind.WHOLE),float(pattern.kind is SplitKind.Y_SPLIT),
                float(pattern.kind is SplitKind.X_SPLIT),float(pattern.mandatory),pattern.t if pattern.t is not None else .5,
                float(pattern.rail is Rail.UPPER),float(pattern.rail is Rail.LOWER),len(children)/2,
                min(lengths)/sum(lengths),max(lengths)/sum(lengths),min(processing)/2500,max(processing)/2500])
            mode_plans=[]
            for after,actions,indices,features in branches:
                plan=dict(parent_id=parent,pattern=pattern,state=after,actions=actions,
                    child_indices=torch.stack(indices),child_features=torch.stack(features),parent_features=pf,mode_features=mode)
                if remaining is not None:plan['remaining_eligible_workload']=remaining
                mode_plans.append(plan)
            if mode_plans:
                # Geometry-dependent final features use this mode's native blocks.
                after_rows=parent_action_inputs(graph,state,mode_plans,blocks,config)['after_features']
                for plan,row in zip(mode_plans,after_rows):plan['after_features']=row
                plans.extend(mode_plans)
    return tuple(plans)


def reflected_start_parent_orders(space, state, pending, selected, model, config):
    """Score an empty prefix in equivalent coordinates with the same parent head."""

    _, blocks, patterns = complete_parent_view(space, selected)
    if any(state.routes):
        raise ValueError('reflected startup ranking requires empty routes')
    graph = space['parent_proposal_graph'][1]
    ids, features = joint_parent_inputs(state, pending, blocks, patterns, config,
        include_opportunities=model.opportunity_features, filter_legal_parents=False)
    orders = []
    for flip_x, flip_y in ((True, False), (False, True), (True, True)):
        robots = [r ^ int(flip_x) ^ (2 * int(flip_y)) for r in range(4)]
        rows = graph.node_features.clone()
        welds = rows[:graph.robot_offset]
        if flip_x:
            welds[:, [2, 4, 6]] = sum(config.workspace_x) / 20 - welds[:, [2, 4, 6]]
            welds[:, [8, 11]] *= -1
        if flip_y:
            welds[:, [3, 5, 7]] = sum(config.workspace_y) / 12 - welds[:, [3, 5, 7]]
            welds[:, [9, 12]] *= -1
        welds[:, 14:18] = welds[:, [14 + r for r in robots]]
        nodes = torch.arange(len(rows))
        nodes[graph.robot_offset:] = graph.robot_offset + torch.tensor(robots)
        reflected = replace(graph, node_features=rows, edge_index=nodes[graph.edge_index])
        columns = list(range(features.shape[1]))
        for start in (0, 4, *range(12, len(columns), 4)):
            columns[start:start + 4] = [start + r for r in robots]
        embeddings = model.contextualize_state(reflected, model.encode_static(reflected), state.routes, state.directions)
        logits = model.parent_logits(reflected, embeddings, ids, features[:, columns]).tolist()
        orders.append(tuple(p for _, p in sorted(zip((-value for value in logits), ids))))
    return tuple(orders)


def route_distance_parent_orders(space, state, selected, gnn_order, config, *, gaps):
    """Rank pending parents by distance from the current directed route for each robot."""
    from mrta_reference.geometry import oriented_endpoints
    from mrta_search.initialization import bounded_insertion_positions
    from mrta_search.lns import has_joint_block_candidate

    _, blocks, patterns = complete_parent_view(space, selected)
    routes = tuple(tuple(oriented_endpoints(blocks[b], d) for b, d in zip(route, directions))
                   for route, directions in zip(state.routes, state.directions))
    distances = [{} for _ in routes]
    for parent in gnn_order:
        pattern = patterns[parent]
        for child in space['options'][pattern]:
            for robot in _eligible_robots(child, config):
                if not has_joint_block_candidate(state, child, pattern, blocks, config, robot_id=robot):
                    continue
                route = routes[robot]
                ends = child.start, child.end
                if route:
                    distance = min(math.dist(route[-1][1], point) for point in ends)
                    if gaps:
                        distance = min(distance, *(math.dist(point, route[0][0]) for point in ends))
                        for position in bounded_insertion_positions(len(route), 8)[1:-1]:
                            before, after = route[position - 1][1], route[position][0]
                            detour = min(math.dist(before, a) + math.dist(b, after) for a, b in (ends, ends[::-1]))
                            distance = min(distance, detour - math.dist(before, after))
                else:
                    boundary = config.workspace_x[robot % 2]
                    distance = min(abs(point[0] - boundary) for point in ends)
                distances[robot][parent] = min(distance, distances[robot].get(parent, math.inf))
    return tuple(tuple(sorted((p for p in gnn_order if p in costs), key=costs.__getitem__)) for costs in distances)


def supplement_parent_ids(gnn_order, supplemental_orders, limit):
    chosen = list(gnn_order[:limit // 2])
    selected = set(chosen)
    rankings = [iter(order) for order in supplemental_orders]
    while len(chosen) < limit and rankings:
        for ranking in rankings.copy():
            parent = next((p for p in ranking if p not in selected), None)
            if parent is None:
                rankings.remove(ranking)
                continue
            chosen.append(parent)
            selected.add(parent)
            if len(chosen) == limit:
                break
    return tuple((chosen + [p for p in gnn_order if p not in selected])[:limit])


def route_progress_supplements(space, state, pending, selected, model, config, gnn_order):
    """Use one parent head and geometric supplements for the current route stage."""
    route_lengths = tuple(map(len, state.routes))
    if not any(route_lengths):
        return reflected_start_parent_orders(space, state, pending, selected, model, config)
    if min(route_lengths) < 3:
        shortest = sorted(range(4), key=lambda r: (route_lengths[r], r))[:2]
        ends = route_distance_parent_orders(space, state, selected, gnn_order, config, gaps=False)
        return tuple(ends[r] for r in shortest)
    if 3 * len(pending) <= len(space['parents']):
        return route_distance_parent_orders(space, state, selected, gnn_order, config, gaps=True)
    ends = route_distance_parent_orders(space, state, selected, gnn_order, config, gaps=False)
    whole = tuple(p for p in gnn_order if space['catalog'][p][0].kind is SplitKind.WHOLE)
    return (*ends, whole)


def route_progress_complete_candidates(space, state, pending, selected, model, config, *,
                                       parent_limit=16, first_limit=16, second_limit=8):
    """Generate exactly the measured whole-parent pool, with the full bounded WHOLE variants."""
    order = gnn_parent_orders(space, state, pending, selected, model, config)['gnn']
    supplements = route_progress_supplements(space, state, pending, selected, model, config, order)
    ids = supplement_parent_ids(order, supplements, parent_limit)
    return complete_parent_candidates(space, state, pending, selected, config,
        first_limit=first_limit, second_limit=second_limit, candidate_parent_ids=ids,
        include_remaining_workload=True, remaining_after_parent=True)


def complete_parent_inputs(plans):
    """Collate exactly the same complete-action pool for training and decoding."""
    n=len(plans);dim=plans[0]['child_features'].shape[1]
    indices=torch.full((n,2,4),-1,dtype=torch.long);features=torch.zeros(n,2,dim);mask=torch.zeros(n,2)
    for i,p in enumerate(plans):
        k=len(p['actions']);indices[i,:k]=p['child_indices'];features[i,:k]=p['child_features'];mask[i,:k]=1
    result=dict(parent_ids=tuple(p['parent_id'] for p in plans),
        parent_features=torch.stack([p['parent_features'] for p in plans]),child_indices=indices,
        child_features=features,child_mask=mask,after_features=torch.stack([p['after_features'] for p in plans]),
        mode_features=torch.stack([p['mode_features'] for p in plans]))
    if 'remaining_eligible_workload' in plans[0]:
        result['remaining_eligible_workload']=torch.stack([p['remaining_eligible_workload'] for p in plans])
    return result


def construct_complete_parent_gnn(parents,model,config,*,seed=None,space=None,state=None,selected=None,
                                  pending=None,observer=None,forced_plan=None,parent_limit=None,
                                  parent_strategy=None,proposal_graph_mode='native',remaining_after_parent=False,
                                  first_limit=None,second_limit=None):
    """Empty routes -> joint actions -> complete solution, with no other solver."""
    from .lns import oriented_route_state
    started=time.perf_counter();actions=[]
    stored_parent_limit = getattr(model, 'compact_parent_limit', None)
    if parent_limit is None:
        parent_limit = stored_parent_limit
    # The saved 16-parent branch uses the measured route-progress pool.
    if parent_strategy is None and stored_parent_limit == 16:
        parent_strategy = 'route_progress'
    if first_limit is None:
        first_limit = 16 if parent_strategy == 'route_progress' else 8
    if second_limit is None:
        second_limit = 8 if parent_strategy == 'route_progress' else 4
    compact=parent_limit is not None
    space=complete_parent_space(parents,config,compact=compact) if space is None else space
    selected={} if selected is None else dict(selected)
    pending={p.parent_id for p in parents}-set(selected) if pending is None else set(pending)
    _,blocks,_=complete_parent_view(space,selected)
    state=oriented_route_state(((),)*4,((),)*4,blocks,config) if state is None else state
    generator=None if seed is None else torch.Generator().manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        embeddings=model.encode_static(space['graph'])
        while pending:
            options=dict(first_limit=first_limit,second_limit=second_limit,parent_limit=parent_limit,
                         include_remaining_workload=True,remaining_after_parent=remaining_after_parent) if compact else {}
            if parent_strategy == 'route_progress':
                plans = route_progress_complete_candidates(space, state, pending, selected, model, config,
                    parent_limit=parent_limit, first_limit=first_limit, second_limit=second_limit)
            elif parent_strategy is not None:
                orders=gnn_parent_orders(space,state,pending,selected,model,config,proposal_graph_mode=proposal_graph_mode)
                options['candidate_parent_ids']=compact_parent_ids(space,pending,parent_limit=parent_limit,
                    strategy=parent_strategy,orders=orders)
                plans=complete_parent_candidates(space,state,pending,selected,config,**options)
            else:
                plans=complete_parent_candidates(space,state,pending,selected,config,**options)
            if not plans:return JointCompletionResult(None,None,tuple(actions),time.perf_counter()-started,'NO_COMPLETE_PARENT_ACTION',state)
            context=model.contextualize_state(complete_parent_view(space,selected)[0],embeddings,state.routes,state.directions)
            scores=model.parent_action_scores(space['graph'],context,complete_parent_inputs(plans))
            if not torch.isfinite(scores).all():raise ValueError('NONFINITE_COMPLETE_PARENT_SCORE')
            index=int(scores.argmax()) if generator is None else int(torch.multinomial(torch.softmax(scores,0),1,generator=generator))
            best=plans[index] if forced_plan is None else forced_plan
            if forced_plan is not None:
                from .lns import joint_action_key
                match=[p for p in plans if p['pattern']==best['pattern'] and tuple(map(joint_action_key,p['actions']))==tuple(map(joint_action_key,best['actions']))]
                if len(match)!=1:raise ValueError('forced complete action is outside inference pool')
                best=match[0];forced_plan=None
            if observer is not None:observer(space,state,tuple(sorted(pending)),dict(selected),plans,best)
            state=best['state'];selected[best['parent_id']]=best['pattern'];pending.remove(best['parent_id']);actions.extend(best['actions'])
        solution=canonicalize(space['parents'],tuple(selected[p.parent_id] for p in space['parents']),
            tuple(Route(r,state.routes[r]) for r in range(4)),config,scope=FORMAL_SCOPE_V2)
        if tuple(r.block_ids for r in solution.routes)!=state.routes:
            raise ValueError('complete joint action unexpectedly changed under canonicalization')
        return JointCompletionResult(solution,state.directions,tuple(actions),time.perf_counter()-started,oriented_state=state)


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

