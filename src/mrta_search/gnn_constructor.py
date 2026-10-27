"""Block-level GNN policy for fixed-split task assignment.

This module deliberately does not depend on the historical candidate rankers.  The
GAT encoder sees the actual welding blocks produced by the frozen first-round split
rule, and the policy head scores eligible robots while an assignment is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
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
            self.layer1(graph.node_features, graph.edge_index, graph.edge_type)
        )
        return self.layer2(hidden, graph.edge_index, graph.edge_type)

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
            self.layer1(graph.node_features, graph.edge_index, graph.edge_type)
        )
        return self.layer2(hidden, graph.edge_index, graph.edge_type)

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

    def __init__(self, hidden_dim=64, heads=4, *, normalize_inputs=False):
        super().__init__(hidden_dim, heads)
        self.hidden_dim, self.heads = hidden_dim, heads
        self.policy = nn.Identity()  # Assignment-only head is not used here.
        self.parent_head = nn.Sequential(nn.Linear(hidden_dim+JOINT_PARENT_DIM,hidden_dim),
                                         nn.ReLU(),nn.Linear(hidden_dim,1))
        self.insertion_head = nn.Sequential(nn.Linear(4*hidden_dim+JOINT_DYNAMIC_DIM,hidden_dim),
                                            nn.ReLU(),nn.Linear(hidden_dim,1))
        # Optional TRAIN-fitted scaling; the default preserves V1 checkpoints
        # and logits exactly. No new direction head or encoder is introduced.
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            for prefix,dim in (('joint',JOINT_DYNAMIC_DIM),('parent',JOINT_PARENT_DIM)):
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

    def scale_inputs(self, features, prefix):
        if self.normalize_inputs:
            return (features-getattr(self,prefix+'_mean'))/getattr(self,prefix+'_scale')
        return features

    def parent_logits(self, graph, embeddings, parent_ids, features):
        pooled = torch.stack([embeddings[[graph.block_index[b.block_id] for b in graph.blocks
                                          if b.parent_id==p]].mean(0) for p in parent_ids])
        return self.parent_head(torch.cat((pooled,self.scale_inputs(features,'parent')),dim=1)).squeeze(1)

    def joint_action_logits(self, embeddings, indices, features):
        context = embeddings[indices.clamp_min(0)] * (indices>=0).unsqueeze(-1)
        return self.insertion_head(torch.cat((context.flatten(1),self.scale_inputs(features,'joint')),dim=1)).squeeze(1)


@dataclass(frozen=True)
class JointCompletionResult:
    solution: CanonicalSolution | None
    directions: tuple[tuple[int, ...], ...] | None
    actions: tuple
    inference_seconds: float
    rejection_reason: str | None = None


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


def joint_parent_inputs(state, pending, blocks, patterns, config, *, insertion_limit=8):
    from .lns import joint_block_candidates
    scale = max(1.0,sum(config.process_time(b.length) for b in blocks.values()))
    travel = [sum(arcs) for arcs in state.arc_costs]
    ids, features = [], []
    for p in sorted(pending):
        children = sorted((b for b in blocks.values() if b.parent_id==p),key=lambda b:b.block_id)
        if not joint_block_candidates(state,children[0],patterns[p],blocks,config,
                                       insertion_limit=insertion_limit,first_only=True):
            continue
        legal = [r for r in range(4) if robot_is_eligible(children[0],r,config)]
        if patterns[p].kind is SplitKind.X_SPLIT:
            legal = [0 if patterns[p].rail is Rail.UPPER else 2]
        ids.append(p)
        features.append([*(v/scale for v in state.loads),*(v/scale for v in travel),
                         sum(config.process_time(b.length) for b in children)/scale,
                         min(state.loads[r]+travel[r] for r in legal)/scale,
                         len(pending)/max(1,len(patterns)),len(children)/2])
    return tuple(ids),torch.tensor(features,dtype=torch.float32).reshape(-1,JOINT_PARENT_DIM)


def joint_candidate_inputs(graph, state, variants, blocks, config):
    """Only geometry, the current prefix and executable local next states."""
    from mrta_reference.geometry import oriented_endpoints
    scale = max(1.0,sum(config.process_time(b.length) for b in blocks.values()))
    n = max(1,len(blocks))
    old_travel = [sum(arcs) for arcs in state.arc_costs]
    def overlap(s,r):
        left,right = (r//2*2,r//2*2+1)
        if not s.routes[left] or not s.routes[right]:
            return 0.0
        left_end = max(max(blocks[b].start[0],blocks[b].end[0]) for b in s.routes[left])
        right_start = min(min(blocks[b].start[0],blocks[b].end[0]) for b in s.routes[right])
        return max(0.0,left_end+config.interference_dx-right_start)/20
    def first_x(s,r):
        return (oriented_endpoints(blocks[s.routes[r][0]],s.directions[r][0])[0][0]/20
                if s.routes[r] else 0.0)
    indices, features = [], []
    for after,action in variants:
        r,k = action.robot,action.position
        route = state.routes[r]
        pre = route[k-1] if k else None
        post = route[k] if k<len(route) else None
        indices.append([graph.block_index[action.block_id],graph.robot_offset+r,
                        graph.block_index[pre] if pre else -1,graph.block_index[post] if post else -1])
        start,end = oriented_endpoints(blocks[action.block_id],action.direction)
        pre_old = state.directions[r][k-1] if pre else 0
        post_old = state.directions[r][k] if post else 0
        pre_new = after.directions[r][k-1] if pre else 0
        post_new = after.directions[r][k+1] if post else 0
        incoming = (math.dist(oriented_endpoints(blocks[pre],pre_new)[1],start)/config.empty_speed if pre else 0)
        outgoing = (math.dist(end,oriented_endpoints(blocks[post],post_new)[0])/config.empty_speed if post else 0)
        old_arc = (math.dist(oriented_endpoints(blocks[pre],pre_old)[1],
                             oriented_endpoints(blocks[post],post_old)[0])/config.empty_speed if pre and post else 0)
        estimated = [p+sum(arcs) for p,arcs in zip(after.loads,after.arc_costs)]
        features.append([*(v/scale for v in state.loads),*(v/scale for v in old_travel),
                         *(v/scale for v in estimated),*(len(route)/n for route in state.routes),
                         k/max(1,len(route)),float(pre is not None),float(post is not None),float(action.direction),
                         pre_old,pre_new,post_old,post_new,len(action.neighbour_changes)/2,
                         incoming/scale,outgoing/scale,old_arc/scale,(after.travel-state.travel)/scale,
                         overlap(state,r),overlap(after,r),float(r%2==0),
                         start[0]/20,start[1]/12,end[0]/20,end[1]/12,
                         first_x(state,r),first_x(after,r),first_x(state,r^1),max(estimated)/scale,sum(estimated)/scale])
    return torch.tensor(indices,dtype=torch.long).reshape(-1,4),torch.tensor(features,dtype=torch.float32).reshape(-1,JOINT_DYNAMIC_DIM)


def prepare_joint_teacher_steps(reference, routes, directions, pending, actions, config, *, insertion_limit=8):
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
        ids,parent_features = joint_parent_inputs(state,pending,blocks,patterns,config,insertion_limit=insertion_limit)
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
            indices,features = joint_candidate_inputs(graph,state,variants,blocks,config)
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


def construct_or_repair_with_gnn(reference, routes, directions, pending_parent_ids,
                                 model: JointInsertionGNN, config, *, insertion_limit=8):
    """One sequential neural policy, usable for partial or four-empty routes.

    It returns failure as failure. No scheduler, heuristic repair or direction
    optimizer is called; complete results are scheduled once by the caller.
    """
    from .lns import joint_block_candidates
    started = time.perf_counter()
    actions = []
    try:
        graph,blocks,patterns = joint_graph(reference,config)
        state = joint_start_state(reference,routes,directions,pending_parent_ids,blocks,config)
        pending = set(pending_parent_ids)
        model.eval()
        with torch.inference_mode():
            embeddings = model.encode_static(graph)
            while pending:
                ids,features = joint_parent_inputs(state,pending,blocks,patterns,config,insertion_limit=insertion_limit)
                if not ids:
                    raise ValueError('NO_LEGAL_PARENT_ACTION')
                parent_logits = model.parent_logits(graph,embeddings,ids,features)
                if not torch.isfinite(parent_logits).all():
                    raise ValueError('NONFINITE_PARENT_LOGITS')
                parent = ids[int(torch.argmax(parent_logits))]
                children = sorted((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
                for child in children:
                    variants = joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=insertion_limit)
                    if not variants:
                        raise ValueError('NO_LEGAL_CHILD_INSERTION')
                    indices,dynamic = joint_candidate_inputs(graph,state,variants,blocks,config)
                    logits = model.joint_action_logits(embeddings,indices,dynamic)
                    if not torch.isfinite(logits).all():
                        raise ValueError('NONFINITE_INSERTION_LOGITS')
                    state,action = variants[int(torch.argmax(logits))]
                    actions.append(action)
                pending.remove(parent)
        solution = canonicalize(reference.parents,reference.patterns,
                                tuple(Route(r,state.routes[r]) for r in range(4)),config,scope=FORMAL_SCOPE_V2,
                                revision=reference.revision+1)
        if solution.patterns!=reference.patterns or tuple(r.block_ids for r in solution.routes)!=state.routes:
            raise ValueError('FIXED_PATTERN_CHANGED')
        return JointCompletionResult(solution,state.directions,tuple(actions),time.perf_counter()-started)
    except (ValueError,KeyError,IndexError,ArithmeticError) as error:
        return JointCompletionResult(None,None,tuple(actions),time.perf_counter()-started,str(error))


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

