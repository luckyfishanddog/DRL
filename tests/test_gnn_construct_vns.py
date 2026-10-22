from __future__ import annotations

import random

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
    construct_solution,
    fixed_blocks,
    fixed_split_patterns,
    solution_from_assignment,
)
from mrta_search.vns import (
    GLOBAL_BALANCED,
    LEGACY_LOCAL_ABSOLUTE,
    n1_relocate_candidates,
    n2_swap_candidates,
    n3_route_candidates,
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

