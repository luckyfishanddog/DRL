from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gnn_construct_vns_pilot as pilot_script

from mrta_baselines.common import CommonBaselineEvaluator
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import (
    OperationKind,
    ParentWeld,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.scheduler import build_operation_templates, build_robot_routes
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_search.gnn_constructor import (
    BlockAssignmentGNN,
    BlockAssignmentRankingModel,
    build_block_robot_graph,
    construct_ranked_solution,
    construct_solution,
    dag_safe_ordered_routes,
    dynamic_features,
    fixed_blocks,
    fixed_split_patterns,
    rule_assignment,
    solution_from_assignment,
)
from mrta_search.vns import (
    GLOBAL_BALANCED,
    LEGACY_LOCAL_ABSOLUTE,
    n1_relocate_candidates,
    n2_swap_candidates,
    n3_route_candidates,
    evaluate_initial_with_shared_repair,
    run_vns,
)


CONFIG = ScientificConfig()


def _parents():
    return (
        ParentWeld("mandatory", (4.0, 4.0), (4.0, 8.0)),
        ParentWeld("shared-a", (5.0, 6.0), (7.0, 6.0)),
        ParentWeld("shared-b", (12.0, 6.0), (14.0, 6.0)),
        ParentWeld("upper", (2.0, 9.0), (3.0, 9.0)),
        ParentWeld("lower", (16.0, 3.0), (17.0, 3.0)),
    )


def _solution():
    return construct_solution(_parents(), CONFIG).solution


def _imbalanced_solution():
    parents = (
        ParentWeld("a", (1.0, 6.0), (4.5, 6.0)),
        ParentWeld("b", (5.0, 6.0), (8.5, 6.0)),
        ParentWeld("c", (9.0, 6.0), (12.5, 6.0)),
        ParentWeld("d", (13.0, 6.0), (16.5, 6.0)),
        ParentWeld("e", (2.0, 6.0), (2.5, 6.0)),
        ParentWeld("f", (3.0, 6.0), (3.5, 6.0)),
        ParentWeld("g", (10.0, 6.0), (10.5, 6.0)),
        ParentWeld("h", (15.0, 6.0), (15.5, 6.0)),
    )
    patterns = tuple(
        SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents
    )
    solution = canonicalize(
        parents,
        patterns,
        {
            0: ("a::whole", "c::whole", "b::whole", "d::whole"),
            1: ("e::whole", "f::whole"),
            2: ("g::whole",),
            3: ("h::whole",),
        },
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    return parents, solution


def _assert_assignment_legal(solution):
    blocks = block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2)
    actual = []
    for route in solution.routes:
        for block_id in route.block_ids:
            actual.append(block_id)
            assert robot_is_eligible(blocks[block_id], route.robot_id, CONFIG)
    assert len(actual) == len(set(actual)) == len(blocks)
    assert set(actual) == set(blocks)


def test_fixed_blocks_only_allow_eligible_robots_and_shared_blocks_allow_both_rails():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    upper = next(block for block in blocks if block.parent_id == "upper")
    shared = next(block for block in blocks if block.parent_id == "shared-a")
    assert [robot_is_eligible(upper, robot, CONFIG) for robot in range(4)] == [True, True, False, False]
    assert all(robot_is_eligible(shared, robot, CONFIG) for robot in range(4))


def test_mandatory_y_split_children_are_distinct_complete_and_not_whole():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    mandatory = next(pattern for pattern in patterns if pattern.parent_id == "mandatory")
    assert mandatory.kind is SplitKind.Y_SPLIT and mandatory.mandatory
    solution = _solution()
    assigned = [item for route in solution.routes for item in route.block_ids]
    assert assigned.count("mandatory::0") == 1
    assert assigned.count("mandatory::1") == 1
    assert "mandatory::whole" not in assigned


def test_gnn_constructor_covers_every_block_once_with_only_legal_actions():
    model = BlockAssignmentGNN()
    result = construct_solution(_parents(), CONFIG, model=model)
    assert result.legal_action_fraction == 1.0
    _assert_assignment_legal(result.solution)


def test_ranking_model_direct_constructor_is_deterministic_complete_and_legal():
    import torch

    torch.manual_seed(71)
    model = BlockAssignmentRankingModel()
    first = construct_ranked_solution(
        _parents(), CONFIG, model, assignment_kind="MODEL"
    )
    second = construct_ranked_solution(
        _parents(), CONFIG, model, assignment_kind="MODEL"
    )
    assert first.solution.canonical_json == second.solution.canonical_json
    _assert_assignment_legal(first.solution)


def test_weld_edge_ablation_removes_only_weld_relations():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    full = build_block_robot_graph(blocks, patterns, CONFIG)
    ablated = build_block_robot_graph(
        blocks, patterns, CONFIG, include_weld_edges=False
    )
    assert any(int(kind) in (1, 2) for kind in full.edge_type)
    assert not any(int(kind) in (1, 2) for kind in ablated.edge_type)
    assert any(int(kind) == 3 for kind in ablated.edge_type)
    assert ablated.node_features.equal(full.node_features)


def test_joint_assignment_and_pairwise_ranking_heads_both_receive_gradients():
    import torch

    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    graph = build_block_robot_graph(blocks, patterns, CONFIG)
    by_id = {block.block_id: block for block in blocks}
    assignment = rule_assignment(blocks, CONFIG)
    routes = [[] for _ in range(4)]
    model = BlockAssignmentRankingModel()
    embeddings = model.encode_static(graph)
    first = blocks[0]
    rows = dynamic_features(first, routes, by_id, CONFIG, 0, len(blocks))
    mask = torch.tensor(
        [robot_is_eligible(first, robot, CONFIG) for robot in range(4)],
        dtype=torch.bool,
    )
    target = dict(assignment)[first.block_id]
    assignment_loss = torch.nn.functional.cross_entropy(
        model.action_logits(graph, embeddings, first.block_id, rows, mask).unsqueeze(0),
        torch.tensor([target]),
    )
    same_robot = [block_id for block_id, robot in assignment if robot == target]
    if len(same_robot) < 2:
        same_robot = [block.block_id for block in blocks[:2]]
        target = next(
            robot
            for robot in range(4)
            if all(robot_is_eligible(by_id[item], robot, CONFIG) for item in same_robot)
        )
    score0 = model.ordering_score(
        graph,
        embeddings,
        same_robot[0],
        target,
        dynamic_features(by_id[same_robot[0]], routes, by_id, CONFIG, 0, len(blocks)),
    )
    score1 = model.ordering_score(
        graph,
        embeddings,
        same_robot[1],
        target,
        dynamic_features(by_id[same_robot[1]], routes, by_id, CONFIG, 0, len(blocks)),
    )
    (assignment_loss + torch.nn.functional.softplus(-(score0 - score1))).backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.assignment_head.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.ranking_head.parameters()
    )


def test_dag_safe_constructor_rejects_precedence_cycle():
    import torch

    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    graph = build_block_robot_graph(blocks, patterns, CONFIG)
    model = BlockAssignmentRankingModel(encoder_kind="mlp")
    with torch.inference_mode():
        embeddings = model.encode_static(graph)
        assignment = rule_assignment(blocks, CONFIG)
        first, second = blocks[0].block_id, blocks[1].block_id
        try:
            dag_safe_ordered_routes(
                model,
                graph,
                embeddings,
                assignment,
                CONFIG,
                predecessors={first: (second,), second: (first,)},
            )
        except ValueError as error:
            assert "cycle" in str(error)
        else:
            raise AssertionError("cyclic precedence was accepted")


def test_solution_constructor_rejects_an_illegal_robot_before_canonicalization():
    parents = _parents()
    patterns = fixed_split_patterns(parents, CONFIG)
    blocks = fixed_blocks(parents, patterns, CONFIG)
    assignment = []
    for block in blocks:
        robot = 2 if block.parent_id == "upper" else next(
            robot for robot in range(4) if robot_is_eligible(block, robot, CONFIG)
        )
        assignment.append((block.block_id, robot))
    try:
        solution_from_assignment(parents, patterns, blocks, assignment, CONFIG)
    except ValueError as error:
        assert "illegal assignment" in str(error)
    else:
        raise AssertionError("illegal upper-to-lower assignment was accepted")


def test_vns_relocate_and_swap_keep_assignment_legal():
    solution = _solution()
    relocate = n1_relocate_candidates(solution, CONFIG, random.Random(7), cap=10_000)
    swap = n2_swap_candidates(solution, CONFIG, random.Random(7), cap=10_000)
    assert relocate
    assert swap
    for candidate in (*relocate, *swap):
        _assert_assignment_legal(candidate)


def test_route_neighborhood_preserves_each_robot_task_set():
    solution = _solution()
    candidates = n3_route_candidates(solution, CONFIG, random.Random(9), cap=20)
    assert candidates
    expected = [set(route.block_ids) for route in solution.routes]
    signatures = set()
    for candidate in candidates:
        assert [set(route.block_ids) for route in candidate.routes] == expected
        signatures.add(candidate.canonical_json)
    assert len(signatures) == len(candidates)


def test_global_n1_and_n2_selection_reaches_the_critical_robot():
    _, solution = _imbalanced_solution()
    for builder in (n1_relocate_candidates, n2_swap_candidates):
        candidate = builder(
            solution,
            CONFIG,
            random.Random(4),
            cap=1,
            selection_policy=GLOBAL_BALANCED,
        )[0]
        changed = [
            robot
            for robot in range(4)
            if candidate.routes[robot].block_ids != solution.routes[robot].block_ids
        ]
        assert 0 in changed
        _assert_assignment_legal(candidate)


def test_n3_global_selection_uses_critical_route_and_improvement_not_absolute_cost():
    _, solution = _imbalanced_solution()
    global_candidate = n3_route_candidates(
        solution,
        CONFIG,
        random.Random(5),
        cap=1,
        selection_policy=GLOBAL_BALANCED,
    )[0]
    legacy_candidate = n3_route_candidates(
        solution,
        CONFIG,
        random.Random(5),
        cap=1,
        selection_policy=LEGACY_LOCAL_ABSOLUTE,
    )[0]
    global_changed = [
        robot
        for robot in range(4)
        if global_candidate.routes[robot].block_ids != solution.routes[robot].block_ids
    ]
    legacy_changed = [
        robot
        for robot in range(4)
        if legacy_candidate.routes[robot].block_ids != solution.routes[robot].block_ids
    ]
    assert global_changed == [0]
    assert legacy_changed == [1]


def test_same_rail_passing_cannot_pass_independent_certification():
    parents = (
        ParentWeld("a", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("b", (10.0, 8.0), (2.0, 8.0)),
    )
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("a::whole",), 1: ("b::whole",)},
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    robot_routes = build_robot_routes(solution, CONFIG, orientations={0: (0,), 1: (0,)})
    operations = tuple(
        operation
        for route in robot_routes
        for operation in build_operation_templates(route, CONFIG)
    )
    completion = tuple(
        max((op.end_time for op in operations if op.robot_id == robot), default=0.0)
        for robot in range(4)
    )
    fake = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        operations,
        max(completion),
        completion,
        directions=((0,), (0,), (), ()),
    )
    report = certify_schedule(solution, fake, CONFIG, scope=FORMAL_SCOPE_V2)
    assert not report.certified
    assert any("same-rail non-passing" in error for error in report.errors)


def test_same_seed_gnn_inference_and_vns_are_reproducible():
    torch_seed = 123
    import torch

    torch.manual_seed(torch_seed)
    model = BlockAssignmentGNN()
    first = construct_solution(_parents(), CONFIG, model=model)
    second = construct_solution(_parents(), CONFIG, model=model)
    assert first.solution.canonical_json == second.solution.canonical_json

    results = []
    for _ in range(2):
        evaluator = CommonBaselineEvaluator(
            _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
        )
        result = run_vns(
            first.solution,
            evaluator,
            CONFIG,
            seed=77,
            per_neighborhood_budget=2,
        )
        results.append(result)
    assert results[0].best is not None and results[1].best is not None
    assert results[0].initial is not None and results[1].initial is not None
    assert results[0].best.solution.canonical_json == results[1].best.solution.canonical_json
    assert results[0].stats.improvements == results[1].stats.improvements
    assert results[0].best.metrics.cmax <= results[0].initial.metrics.cmax
    assert results[0].best.certification is not None
    assert results[0].best.certification.certified


def test_late_initial_certification_is_not_accepted_at_hard_deadline():
    evaluator = CommonBaselineEvaluator(
        _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
    )
    completed = evaluator.evaluate(_solution(), source="ON_TIME_REFERENCE")
    assert completed.status.value == "COMPLETED"

    class LateEvaluator:
        expired = True

        def evaluate(self, solution, *, source):
            return replace(completed, solution=solution, source=source, completed_at=1.01)

    initial, attempts, failures = evaluate_initial_with_shared_repair(
        _solution(), LateEvaluator(), CONFIG, accept_completed_by=1.0
    )
    assert initial is None
    assert attempts == 1
    assert failures == ("LATE_CERTIFIED",)


def test_end_to_end_budget_charges_model_load_construction_and_uses_remaining_time(monkeypatch):
    entry = {
        "instance_id": "unit:tiny",
        "N": 5,
        "split": "MLP_DEV",
    }
    original_construct = pilot_script.construct_solution
    captured = {}

    def delayed_load_model():
        time.sleep(0.02)
        return BlockAssignmentGNN(), {}

    def delayed_construct(parents, config, *, model=None):
        time.sleep(0.02)
        return original_construct(parents, config, model=model)

    def fake_run_vns(initial_solution, evaluator, config, **kwargs):
        captured["time_limit"] = evaluator.time_limit
        captured["accept_completed_by"] = kwargs["accept_completed_by"]
        return pilot_script.VNSResult(
            None, None, pilot_script.VNSStats(), 0, (), evaluator.elapsed
        )

    monkeypatch.setattr(pilot_script.ranker_dataset, "load_parents", lambda _: _parents())
    monkeypatch.setattr(pilot_script, "load_model", delayed_load_model)
    monkeypatch.setattr(pilot_script, "construct_solution", delayed_construct)
    monkeypatch.setattr(pilot_script, "run_vns", fake_run_vns)
    row = pilot_script.run_end_to_end(entry, "GNN", 7, CONFIG, 0.2)
    assert row["timing_scope"] == "END_TO_END_EXCLUDING_INPUT_READ"
    assert row["model_load_seconds"] >= 0.015
    assert row["construction_seconds"] >= 0.015
    assert captured["time_limit"] < 0.17
    assert captured["accept_completed_by"] == captured["time_limit"]


def test_assignment_change_diagnostic_ignores_route_order_only_changes():
    evaluator = CommonBaselineEvaluator(
        _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
    )
    initial = evaluator.evaluate(_solution(), source="INITIAL")
    assert initial.status.value == "COMPLETED"
    reordered = canonicalize(
        initial.solution.parents,
        initial.solution.patterns,
        {
            route.robot_id: tuple(reversed(route.block_ids))
            for route in initial.solution.routes
        },
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    final = replace(initial, solution=reordered)
    diagnostics = pilot_script._knowledge_diagnostics(initial, final, CONFIG)
    assert diagnostics["assignment_changes"] == 0
    assert diagnostics["changed_route_robots"] > 0
    assert diagnostics["pure_order_changed_robots"] > 0


def test_existing_role_manifest_keeps_train_and_dev_disjoint():
    representatives = pilot_script.ranker_dataset.representatives_v05()
    train = {
        entry["instance_id"] for entry in representatives if entry["split"] == "MLP_TRAIN"
    }
    dev = {
        entry["instance_id"] for entry in representatives if entry["split"] == "MLP_DEV"
    }
    assert len(train) == 17
    assert len(dev) == 6
    assert train.isdisjoint(dev)

