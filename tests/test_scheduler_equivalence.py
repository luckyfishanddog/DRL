from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest

import mrta_reference.scheduler as scheduler_module
from mrta_reference.geometry import generate_y_split_patterns, robot_is_eligible
from mrta_reference.model import Operation, OperationKind, ParentWeld, Route, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import (
    _remaining_processing_suffix,
    earliest_safe_start_optimized,
    earliest_safe_start_slow,
    reference_schedule_from_templates_optimized,
    reference_schedule_from_templates_slow,
    reference_schedule_optimized,
    reference_schedule_slow,
    relevant_fixed_operations,
)
from mrta_reference.solution import block_map, canonicalize
from mrta_search import SearchConfig, run_bounded_sa_oi


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _assert_schedule_equivalent(slow: ScheduleResult, optimized: ScheduleResult, config=FAST) -> None:
    assert slow.status is optimized.status
    assert slow.directions == optimized.directions
    assert slow.wait_for_graph == optimized.wait_for_graph
    assert slow.diagnostics == optimized.diagnostics
    if slow.cmax is None:
        assert optimized.cmax is None
    else:
        assert slow.cmax == pytest.approx(optimized.cmax)
    assert slow.robot_completion == pytest.approx(optimized.robot_completion)
    assert len(slow.operations) == len(optimized.operations)
    for left, right in zip(slow.operations, optimized.operations):
        assert left.robot_id == right.robot_id
        assert left.kind is right.kind
        assert left.start == pytest.approx(right.start)
        assert left.end == pytest.approx(right.end)
        assert left.start_time == pytest.approx(right.start_time)
        assert left.end_time == pytest.approx(right.end_time)
        assert left.sequence_index == right.sequence_index
        assert left.block_id == right.block_id
        assert left.operation_id == right.operation_id


def _move_templates(robot_id: int, points):
    operations = []
    cursor = 0.0
    for index, (start, end) in enumerate(zip(points, points[1:])):
        duration = math.dist(start, end)
        operations.append(
            Operation(
                f"R{robot_id}:{index}",
                robot_id,
                OperationKind.MOVE,
                cursor,
                cursor + duration,
                start,
                end,
                index,
            )
        )
        cursor += duration
    return tuple(operations)


def _oracle_cases():
    return (
        {
            0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            2: _move_templates(2, ((0.0, 6.0), (1.0, 6.0), (2.0, 6.0))),
        },
        {
            0: _move_templates(0, ((2.0, 6.0), (2.0, 1.0), (4.0, 6.0), (3.0, 4.0))),
            2: _move_templates(2, ((1.0, 4.0), (5.0, 4.0), (1.0, 1.0), (2.0, 4.0))),
        },
        {
            0: _move_templates(0, ((4.0, 2.0), (4.0, 5.0), (1.0, 0.0), (6.0, 3.0))),
            2: _move_templates(2, ((0.0, 2.0), (0.0, 0.0), (3.0, 3.0), (0.0, 4.0))),
        },
        {
            0: _move_templates(0, ((5.0, 0.0), (3.0, 4.0), (1.0, 1.0), (1.0, 0.0))),
            2: _move_templates(2, ((4.0, 1.0), (5.0, 0.0), (3.0, 0.0), (1.0, 0.0))),
        },
    )


def test_expired_filter_closed_boundary_and_epsilon_edges() -> None:
    ready = 10.0
    epsilon = FAST.numeric_epsilon

    def fixed(name, end):
        return Operation(name, 1, OperationKind.WAIT, end - 1.0, end, (1.0, 1.0), (1.0, 1.0), 0)

    rows = (
        fixed("equal", ready),
        fixed("minus-half-eps", ready - 0.5 * epsilon),
        fixed("plus-eps", ready + epsilon),
        fixed("expired", ready - 100.0 * epsilon),
        Operation("zero-at-ready", 1, OperationKind.WAIT, ready, ready, (1.0, 1.0), (1.0, 1.0), 0),
    )
    relevant = relevant_fixed_operations(rows, 0, ready, FAST)
    assert {item.operation_id for item in relevant} == {
        "equal",
        "minus-half-eps",
        "plus-eps",
        "zero-at-ready",
    }


def test_interval_sweep_matches_legacy_for_nested_touching_and_random_cases() -> None:
    rng = random.Random(20260928)
    for case in range(500):
        robot = rng.randrange(4)
        ready = rng.uniform(0.0, 10.0)
        duration = rng.choice((0.0, rng.uniform(0.01, 5.0)))
        start = (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
        end = start if duration == 0.0 else (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
        template = Operation(f"candidate-{case}", robot, OperationKind.MOVE, 0.0, duration, start, end, 0)
        fixed = []
        for index in range(rng.randrange(0, 12)):
            other = rng.randrange(4)
            begin = rng.uniform(0.0, 15.0)
            fixed_duration = rng.choice((0.0, rng.uniform(0.01, 5.0)))
            point_a = (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
            point_b = point_a if fixed_duration == 0.0 else (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
            fixed.append(Operation(f"fixed-{case}-{index}", other, OperationKind.MOVE, begin, begin + fixed_duration, point_a, point_b, index))
        slow = earliest_safe_start_slow(template, ready, fixed, FAST)
        relevant = relevant_fixed_operations(fixed, robot, ready, FAST)
        optimized = earliest_safe_start_optimized(template, ready, relevant, FAST)
        if math.isfinite(slow[0]):
            assert optimized[0] == pytest.approx(slow[0])
        else:
            assert math.isinf(optimized[0])
        assert optimized[1:] == slow[1:]


def test_interval_sweep_explicit_nested_touching_identical_and_infinite(monkeypatch) -> None:
    template = Operation(
        "candidate", 0, OperationKind.MOVE, 0.0, 1.0, (0.0, 0.0), (1.0, 0.0), 0
    )
    fixed = tuple(
        Operation(
            name,
            1,
            OperationKind.WAIT,
            0.0,
            20.0,
            (10.0, 10.0),
            (10.0, 10.0),
            index,
        )
        for index, name in enumerate(("outer", "nested", "chain", "identical"))
    )
    intervals = {
        "outer": ((0.0, 2.0),),
        "nested": ((0.0, 1.0),),
        "chain": ((1.5, 4.0),),
        "identical": ((1.5, 4.0),),
    }

    def projected(*args):
        operation = args[4]
        return intervals[operation.operation_id]

    monkeypatch.setattr(scheduler_module, "forbidden_start_intervals", projected)
    slow = earliest_safe_start_slow(template, 0.0, fixed, FAST)
    optimized = earliest_safe_start_optimized(template, 0.0, fixed, FAST)
    assert slow == optimized
    assert slow[0] > 4.0
    assert slow[1] == (1,)
    assert slow[2] == ("chain", "identical", "nested", "outer")

    monkeypatch.setattr(
        scheduler_module,
        "forbidden_start_intervals",
        lambda *args: ((0.0, math.inf),),
    )
    slow_infinite = earliest_safe_start_slow(template, 0.0, fixed[:1], FAST)
    optimized_infinite = earliest_safe_start_optimized(
        template, 0.0, fixed[:1], FAST
    )
    assert slow_infinite == optimized_infinite
    assert math.isinf(slow_infinite[0])


def test_remaining_processing_suffix_excludes_move_and_matches_legacy_sum() -> None:
    templates = {robot: () for robot in range(4)}
    templates[0] = (
        Operation("setup", 0, OperationKind.SETUP, 0.0, 2.0, (0.0, 0.0), (0.0, 0.0), 0),
        Operation("move", 0, OperationKind.MOVE, 2.0, 7.0, (0.0, 0.0), (5.0, 0.0), 1),
        Operation("weld", 0, OperationKind.WELD, 7.0, 10.0, (5.0, 0.0), (8.0, 0.0), 2),
        Operation("post", 0, OperationKind.POST, 10.0, 14.0, (8.0, 0.0), (8.0, 0.0), 3),
    )
    suffix = _remaining_processing_suffix(templates)[0]
    assert suffix == pytest.approx((9.0, 7.0, 7.0, 4.0, 0.0))


@pytest.mark.parametrize("templates", _oracle_cases())
def test_phase2a_e1_e4_template_differential(templates) -> None:
    slow = reference_schedule_from_templates_slow(templates, FAST)
    optimized = reference_schedule_from_templates_optimized(templates, FAST)
    _assert_schedule_equivalent(slow, optimized)


def test_deadlock_infeasible_and_numeric_failure_differential(monkeypatch) -> None:
    parents = (
        ParentWeld("left-to-right", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("right-to-left", (10.0, 8.0), (2.0, 8.0)),
    )
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("left-to-right::whole",), 1: ("right-to-left::whole",)},
        FAST,
    )
    slow = reference_schedule_slow(solution, FAST, orientations={0: (0,), 1: (0,)})
    optimized = reference_schedule_optimized(solution, FAST, orientations={0: (0,), 1: (0,)})
    assert slow.status is ScheduleStatus.DEADLOCK
    _assert_schedule_equivalent(slow, optimized)
    assert slow.wait_for_graph
    assert any("blocked by" in item for item in slow.diagnostics)

    bad = {0: (Operation("bad", 1, OperationKind.WAIT, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0), 0),)}
    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(bad, FAST),
        reference_schedule_from_templates_optimized(bad, FAST),
    )

    valid = {0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0)))}

    def explode(*args, **kwargs):
        raise OverflowError("forced numeric differential")

    monkeypatch.setattr(scheduler_module, "earliest_safe_start_slow", explode)
    slow_numeric = reference_schedule_from_templates_slow(valid, FAST)
    monkeypatch.setattr(scheduler_module, "earliest_safe_start_optimized", explode)
    optimized_numeric = reference_schedule_from_templates_optimized(valid, FAST)
    assert slow_numeric.status is ScheduleStatus.NUMERIC_FAILURE
    _assert_schedule_equivalent(slow_numeric, optimized_numeric)


def _random_solution(rng: random.Random, case: int):
    count = rng.randrange(0, 7)
    parents = []
    patterns = []
    routes = {robot: [] for robot in range(4)}
    for index in range(count):
        y = rng.choice((2.0, 6.0, 10.0))
        x = rng.uniform(0.5, 18.5)
        parent = ParentWeld(f"c{case}-p{index}", (x, y), (x + rng.uniform(0.2, 1.0), y))
        parents.append(parent)
        patterns.append(SplitPattern(parent.parent_id, SplitKind.WHOLE))
    provisional = canonicalize(parents, patterns, {robot: () for robot in range(4)} if not parents else {0: ()}, FAST) if not parents else None
    if parents:
        for parent in parents:
            block_id = f"{parent.parent_id}::whole"
            eligible = [robot for robot in range(4) if (robot < 2 and parent.start[1] >= FAST.by[0]) or (robot >= 2 and parent.start[1] <= FAST.by[1])]
            routes[rng.choice(eligible)].append(block_id)
        for route in routes.values():
            rng.shuffle(route)
        provisional = canonicalize(parents, patterns, routes, FAST)
    directions = {
        robot: tuple(rng.randrange(2) for _ in route.block_ids)
        for robot, route in enumerate(provisional.routes)
    }
    return provisional, directions


def test_hundreds_of_deterministic_random_solution_differentials() -> None:
    rng = random.Random(73013)
    for case in range(300):
        solution, directions = _random_solution(rng, case)
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        formal = scheduler_module.reference_schedule_formal(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert formal.canonical_json() == baseline.canonical_json()
            assert formal.expanded_states == 0


@pytest.mark.parametrize("index", range(4))
def test_formal_e1_e4_policy_and_independent_template_certificate(index):
    from mrta_reference.certifier import certify_template_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V1, FORMAL_SCOPE_V1_1
    templates = _oracle_cases()[index]
    baseline = reference_schedule_from_templates_optimized(templates, FAST)
    historical = scheduler_module.reference_schedule_from_templates_formal(
        templates, FAST, scope=FORMAL_SCOPE_V1
    )
    formal = scheduler_module.reference_schedule_from_templates_formal(
        templates, FAST, scope=FORMAL_SCOPE_V1_1
    )
    if index < 2:
        assert formal.canonical_json() == baseline.canonical_json()
        assert historical.canonical_json() == baseline.canonical_json()
        assert formal.source == "BASELINE" and formal.recovery_rollouts == 0
    elif index == 2:
        assert baseline.status is ScheduleStatus.DEADLOCK
        assert historical.feasible and historical.source == "BOUNDED_DEADLOCK_RECOVERY"
        assert historical.expanded_states == historical.state_budget == 16
        assert historical.cmax == pytest.approx(15.5287634621)
        assert formal.feasible and formal.source == "LIMITED_DISCREPANCY_RECOVERY"
        assert 1 <= formal.recovery_rollouts <= formal.rollout_budget == 32
    else:
        assert historical.status is ScheduleStatus.DEADLOCK
        assert historical.expanded_states == 7 and historical.frontier_exhausted
        assert formal.status is ScheduleStatus.DEADLOCK
        assert formal.cmax is None and formal.recovery_rollouts > 0
        assert formal.frontier_exhausted
    for result in (historical, formal):
        if result.feasible:
            assert certify_template_schedule(templates, result, FAST).certified
            assert not certify_template_schedule(
                templates, replace(result, cmax=result.cmax + 1), FAST
            ).certified
            assert not certify_template_schedule(
                templates, replace(result, operations=result.operations[1:]), FAST
            ).certified


def test_bounded_exhaustion_stays_deadlock_and_numeric_failure_is_not_swallowed(monkeypatch):
    templates = _oracle_cases()[2]
    baseline = reference_schedule_from_templates_optimized(templates, FAST)
    exhausted = scheduler_module._bounded_dispatch_recovery(templates, FAST, baseline, state_budget=1)
    assert exhausted.status is ScheduleStatus.DEADLOCK
    assert exhausted.recovery_exhausted and exhausted.expanded_states == 1
    def explode(*args, **kwargs):
        raise ArithmeticError("forced recovery failure")
    monkeypatch.setattr(scheduler_module, "earliest_safe_start_optimized", explode)
    numeric = scheduler_module._bounded_dispatch_recovery(templates, FAST, baseline, state_budget=16)
    assert numeric.status is ScheduleStatus.NUMERIC_FAILURE
    assert any("forced recovery failure" in d for d in numeric.diagnostics)


def test_bounded_budget_recovery_is_monotonic_non_worsening_and_certified():
    from mrta_reference.certifier import certify_template_schedule
    templates = _oracle_cases()[2]
    baseline = reference_schedule_from_templates_optimized(templates, FAST)
    found = False
    previous_cmax = None
    for budget in (1, 2, 4, 8, 16, 32, 64, 128):
        result = scheduler_module._bounded_dispatch_recovery(
            templates, FAST, baseline, state_budget=budget
        )
        if found:
            assert result.status is ScheduleStatus.FEASIBLE
        if result.feasible:
            found = True
            assert certify_template_schedule(templates, result, FAST).certified
            if previous_cmax is not None:
                assert result.cmax <= previous_cmax + FAST.numeric_epsilon
            previous_cmax = result.cmax
        else:
            assert result.status is ScheduleStatus.DEADLOCK
    assert found


def _known_recoverable_large_case():
    import json
    from pathlib import Path
    from mrta_reference.model import CanonicalSolution

    corpus = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data"
            / "development"
            / "f4_deadlock_stress_corpus.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        row
        for row in corpus["entries"]
        if row["identity"]
        == "328926b7612633797dbe0ae62b62964bbf0a6765b1ef0d6147305b07a0ec6511"
    )
    payload = entry["canonical_solution"]
    solution = CanonicalSolution(
        tuple(ParentWeld(row[0], tuple(row[1]), tuple(row[2])) for row in payload["parents"]),
        tuple(
            SplitPattern(row[0], SplitKind(row[1]), row[2], row[3], row[4])
            for row in payload["patterns"]
        ),
        tuple(Route(row[0], tuple(row[1])) for row in payload["routes"]),
    )
    values = dict(entry["scientific_config"])
    values["workspace_x"] = tuple(values["workspace_x"])
    values["workspace_y"] = tuple(values["workspace_y"])
    config = ScientificConfig(**values)
    directions = tuple(tuple(row) for row in entry["directions"])
    routes = scheduler_module.build_robot_routes(
        solution, config, {robot: directions[robot] for robot in range(4)}
    )
    templates = {
        route.robot_id: scheduler_module.build_operation_templates(route, config)
        for route in routes
    }
    baseline = scheduler_module._optimized_dispatch_outcome(
        templates, config, directions=directions, collect_trace=True
    )
    return solution, config, directions, templates, baseline


def test_dispatch_rollout_trace_is_deterministic_and_snapshot_matches_root_replay():
    from mrta_reference.dispatch_recovery import ForcedDispatchDecision

    templates = _oracle_cases()[2]
    first = scheduler_module._optimized_dispatch_outcome(
        templates, FAST, collect_trace=True
    )
    second = scheduler_module._optimized_dispatch_outcome(
        templates, FAST, collect_trace=True
    )
    assert first.trace == second.trace
    point = first.trace.branch_points[-1]
    decision = ForcedDispatchDecision(
        point.state.identity, point.ordered_choices[1].identity
    )
    snapshot = scheduler_module._optimized_dispatch_outcome(
        templates,
        FAST,
        initial_state=point.state,
        forced_decisions=(decision,),
        collect_trace=True,
    )
    root = scheduler_module._optimized_dispatch_outcome(
        templates, FAST, forced_decisions=(decision,), collect_trace=True
    )
    assert snapshot.result.canonical_json() == root.result.canonical_json()
    assert snapshot.trace.terminal_depth == root.trace.terminal_depth


def test_complete_rollout_engine_reaches_a_404_template_terminal_state():
    points = ((0.0, 10.0), (10.0, 10.0), (0.0, 2.0), (10.0, 2.0))
    templates = {
        robot: tuple(
            Operation(
                f"R{robot}:{index}",
                robot,
                OperationKind.SETUP,
                float(index),
                float(index + 1),
                points[robot],
                points[robot],
                index,
            )
            for index in range(101)
        )
        for robot in range(4)
    }
    outcome = scheduler_module._optimized_dispatch_outcome(
        templates, FAST, collect_trace=True
    )
    assert sum(len(rows) for rows in templates.values()) == 404
    assert outcome.result.status is ScheduleStatus.FEASIBLE
    assert outcome.trace.terminal_depth == 404


def test_limited_discrepancy_recent_branch_rank_order_dedup_and_exact_budget():
    from mrta_reference.dispatch_recovery import (
        ForcedDispatchDecision,
        limited_discrepancy_recovery,
    )

    _, config, directions, templates, baseline = _known_recoverable_large_case()
    observed = []

    def replay(state, decision, causal_blockers):
        point = next(
            branch
            for branch in baseline.trace.branch_points
            if branch.state.identity == state.identity
        )
        rank = next(
            index
            for index, choice in enumerate(point.ordered_choices)
            if choice.identity == decision.choice_identity
        )
        observed.append((point.depth, rank, decision.key, causal_blockers))
        blockers = frozenset(causal_blockers)

        def selector(_state, choices):
            return 1 if blockers and choices[0].operation_id in blockers and len(choices) > 1 else 0

        return scheduler_module._optimized_dispatch_outcome(
            templates,
            config,
            directions=directions,
            initial_state=state,
            forced_decisions=(decision,),
            collect_trace=True,
            choice_selector=selector if blockers else None,
        )

    result = limited_discrepancy_recovery(
        baseline, rollout_budget=4, rollout_from_snapshot=replay
    )
    causal_ids = set(baseline.trace.terminal_blocker_operation_ids)
    causal_points = sorted(
        (
            point
            for point in baseline.trace.branch_points
            if point.ordered_choices[0].operation_id in causal_ids
        ),
        key=lambda point: (-point.depth, point.ordinal, point.identity),
    )
    expected = [
        (point.depth, rank)
        for point in causal_points
        for rank in range(1, len(point.ordered_choices))
    ][:4]
    assert [(depth, rank) for depth, rank, _, _ in observed] == expected
    assert len({decision for _, _, decision, _ in observed}) == len(observed)
    assert result.recovery_rollouts == result.rollout_budget == 4


def test_known_old_2048_case_recovers_with_16_complete_rollouts_monotonically():
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V1_1

    solution, config, directions, templates, baseline = _known_recoverable_large_case()
    rows = []
    budgets = (1, 2, 4, 8, 16, 32, 64, 128)
    for budget in budgets:
        result = scheduler_module._limited_discrepancy_dispatch_recovery(
            templates, config, baseline, rollout_budget=budget
        )
        assert result.rollout_budget == budget
        assert result.recovery_rollouts <= budget
        rows.append(result)
    found = False
    previous = None
    for result in rows:
        if found:
            assert result.status is ScheduleStatus.FEASIBLE
        if result.feasible:
            found = True
            if previous is not None:
                assert result.cmax <= previous + config.numeric_epsilon
            previous = result.cmax
        else:
            assert result.status is ScheduleStatus.DEADLOCK
    assert rows[4].feasible and rows[4].recovery_rollouts <= 16
    formal = scheduler_module.reference_schedule_formal(
        solution,
        config,
        scope=FORMAL_SCOPE_V1_1,
        orientations={robot: directions[robot] for robot in range(4)},
    )
    assert formal.feasible and formal.source == "LIMITED_DISCREPANCY_RECOVERY"
    assert formal.cmax == pytest.approx(rows[4].cmax)
    assert certify_schedule(
        solution, formal, config, scope=FORMAL_SCOPE_V1_1
    ).certified


def test_prepared_v1_1_recovery_is_exactly_equal_to_slow_oracle():
    solution, config, directions, templates, slow_baseline = _known_recoverable_large_case()
    slow = scheduler_module._limited_discrepancy_dispatch_recovery_slow(
        templates, config, slow_baseline, rollout_budget=32
    )
    prepared = scheduler_module.prepare_dispatch_problem(
        templates, config, directions=directions
    )
    fast_baseline = scheduler_module._prepared_dispatch_outcome(
        prepared, collect_trace=True
    )
    fast = scheduler_module._limited_discrepancy_dispatch_recovery_optimized(
        prepared, fast_baseline, rollout_budget=32
    )
    assert fast_baseline.result == slow_baseline.result
    assert fast_baseline.trace == slow_baseline.trace
    assert fast == slow
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V1_1
    formal_fast = replace(
        fast,
        reference_policy_id=FORMAL_SCOPE_V1_1.reference_scheduler_policy_id,
        scope_id=FORMAL_SCOPE_V1_1.scope_id,
        scope_hash=FORMAL_SCOPE_V1_1.scope_hash,
    )
    assert certify_schedule(
        solution, formal_fast, config, scope=FORMAL_SCOPE_V1_1
    ).certified


def test_prepared_branch_index_snapshots_are_isolated_and_replay_identically():
    from mrta_reference.dispatch_recovery import ForcedDispatchDecision

    _, config, directions, templates, _ = _known_recoverable_large_case()
    prepared = scheduler_module.prepare_dispatch_problem(
        templates, config, directions=directions
    )
    baseline = scheduler_module._prepared_dispatch_outcome(prepared, collect_trace=True)
    point = baseline.trace.branch_points[-1]
    decision = ForcedDispatchDecision(
        point.state.identity, point.ordered_choices[1].identity
    )
    first = scheduler_module._prepared_dispatch_outcome(
        prepared,
        initial_state=point.state,
        forced_decisions=(decision,),
        collect_trace=True,
    )
    cache_snapshot = {
        key: tuple(tuple(items) for items in value.operations)
        for key, value in prepared.fixed_index_snapshots.items()
    }
    second = scheduler_module._prepared_dispatch_outcome(
        prepared,
        initial_state=point.state,
        forced_decisions=(decision,),
        collect_trace=True,
    )
    assert first == second
    assert cache_snapshot == {
        key: tuple(tuple(items) for items in value.operations)
        for key, value in prepared.fixed_index_snapshots.items()
    }


def test_v1_1_profile_on_off_has_identical_scientific_result():
    from mrta_reference.scope import FORMAL_SCOPE_V1_1

    solution, config, directions, _, _ = _known_recoverable_large_case()
    without_profile = scheduler_module.reference_schedule_formal(
        solution,
        config,
        scope=FORMAL_SCOPE_V1_1,
        orientations={robot: directions[robot] for robot in range(4)},
    )
    profile = scheduler_module.SchedulerProfile()
    with_profile = scheduler_module.reference_schedule_formal(
        solution,
        config,
        scope=FORMAL_SCOPE_V1_1,
        orientations={robot: directions[robot] for robot in range(4)},
        profile=profile,
    )
    assert with_profile == without_profile
    assert profile.recovery_rollout_count == with_profile.recovery_rollouts
    assert profile.forbidden_interval_cache_hits > 0
    assert profile.peak_index_snapshots > 0


def test_plateau_detector_uses_per_candidate_status_cmax_and_certification(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from profile_scheduler import detect_local_plateaus
    budgets = (16, 32, 64, 128, 256)
    entry = {"replays": {
        str(budget): {
            "status": "FEASIBLE", "Cmax": 10.0, "certified": True
        }
        for budget in budgets
    }}
    assert detect_local_plateaus([entry], budgets) == [16, 32, 64]
    entry["replays"]["64"]["Cmax"] = 9.0
    assert 16 not in detect_local_plateaus([entry], budgets)
    entry["replays"]["64"]["Cmax"] = 10.0
    entry["replays"]["128"]["certified"] = False
    assert 32 not in detect_local_plateaus([entry], budgets)


def test_y_split_empty_route_wait_and_same_rail_boundary_differentials() -> None:
    cases = []
    mandatory = ParentWeld("mandatory", (2.0, 5.0), (2.0, 7.0))
    mandatory_pattern = generate_y_split_patterns(mandatory, FAST)[0]
    cases.append(
        canonicalize(
            (mandatory,),
            (mandatory_pattern,),
            {0: ("mandatory::1",), 2: ("mandatory::0",)},
            FAST,
        )
    )
    optional = ParentWeld("optional", (0.0, 6.0), (4.0, 6.0))
    optional_pattern = generate_y_split_patterns(optional, FAST)[0]
    cases.append(
        canonicalize(
            (optional,),
            (optional_pattern,),
            {0: ("optional::0",), 3: ("optional::1",)},
            FAST,
        )
    )
    for solution in cases:
        directions = {robot: tuple(0 for _ in route.block_ids) for robot, route in enumerate(solution.routes)}
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert scheduler_module.reference_schedule_formal(
                solution, FAST, orientations=directions).canonical_json() == baseline.canonical_json()

    touching = {
        0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        1: _move_templates(1, ((2.5, 0.0), (1.5, 0.0), (3.0, 0.0))),
    }
    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(touching, FAST),
        reference_schedule_from_templates_optimized(touching, FAST),
    )


def test_phase2a_m1_m7_solution_differentials() -> None:
    rows = []

    def whole(parents, routes, directions):
        solution = canonicalize(
            parents,
            tuple(
                SplitPattern(parent.parent_id, SplitKind.WHOLE)
                for parent in parents
            ),
            routes,
            FAST,
        )
        rows.append((solution, directions))

    m1 = (ParentWeld("m1", (1.0, 2.0), (3.0, 2.0)),)
    whole(m1, {2: ("m1::whole",)}, {2: (0,)})

    assignment = (
        ParentWeld("left", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("right", (9.0, 2.0), (10.0, 2.0)),
    )
    whole(
        assignment,
        {2: ("left::whole",), 3: ("right::whole",)},
        {2: (0,), 3: (0,)},
    )
    whole(
        assignment,
        {2: ("left::whole", "right::whole")},
        {2: (0, 0)},
    )

    route = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (5.0, 2.0), (4.0, 2.0)),
        ParentWeld("c", (9.0, 2.0), (10.0, 2.0)),
    )
    whole(
        route,
        {2: ("a::whole", "b::whole", "c::whole")},
        {2: (0, 1, 0)},
    )
    whole(
        route,
        {2: ("a::whole", "c::whole", "b::whole")},
        {2: (0, 0, 1)},
    )
    whole(
        route[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 0)},
    )
    whole(
        route[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 1)},
    )

    optional = ParentWeld("optional-m5", (0.0, 6.0), (4.0, 6.0))
    whole(
        (optional,),
        {0: ("optional-m5::whole",)},
        {0: (0,)},
    )
    optional_pattern = generate_y_split_patterns(optional, FAST)[0]
    optional_solution = canonicalize(
        (optional,),
        (optional_pattern,),
        {0: ("optional-m5::0",), 1: ("optional-m5::1",)},
        FAST,
    )
    rows.append((optional_solution, {0: (0,), 1: (0,)}))

    mandatory = ParentWeld("mandatory-m7", (1.0, 5.0), (1.0, 7.0))
    mandatory_pattern = generate_y_split_patterns(mandatory, FAST)[0]
    mandatory_solution = canonicalize(
        (mandatory,),
        (mandatory_pattern,),
        {0: ("mandatory-m7::1",), 2: ("mandatory-m7::0",)},
        FAST,
    )
    rows.append((mandatory_solution, {0: (0,), 2: (0,)}))

    for solution, directions in rows:
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert scheduler_module.reference_schedule_formal(
                solution, FAST, orientations=directions).canonical_json() == baseline.canonical_json()

    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(_oracle_cases()[1], FAST),
        reference_schedule_from_templates_optimized(_oracle_cases()[1], FAST),
    )


def test_fixed_budget_search_trajectory_is_identical() -> None:
    parents = (
        ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),
        ParentWeld("lower-a", (1.0, 2.0), (3.0, 2.0)),
        ParentWeld("lower-b", (8.0, 2.0), (10.0, 2.0)),
    )
    search_config = SearchConfig(m=42, kdp=6, kref=2, max_iterations=12, time_limit=None)
    slow = run_bounded_sa_oi(parents, FAST, search_config, seed=123, reference_evaluator=reference_schedule_slow)
    optimized = run_bounded_sa_oi(parents, FAST, search_config, seed=123, reference_evaluator=reference_schedule_optimized)
    assert slow.status is optimized.status
    assert slow.best_solution.canonical_hash == optimized.best_solution.canonical_hash
    assert slow.best_directions == optimized.best_directions
    assert slow.best_schedule.cmax == pytest.approx(optimized.best_schedule.cmax)
    assert slow.stats.reference_status_sequence == optimized.stats.reference_status_sequence
    assert slow.stats.proposal_trajectory == optimized.stats.proposal_trajectory
    assert slow.stats.best_improvement_cmax == optimized.stats.best_improvement_cmax


def test_method_independent_sampler_is_deterministic_canonical_and_search_free(
    monkeypatch,
) -> None:
    import profile_scheduler as profiler

    def forbidden(*args, **kwargs):
        raise AssertionError("a search heuristic was called by the direct sampler")

    monkeypatch.setattr(profiler, "run_bounded_sa_oi", forbidden)
    monkeypatch.setattr(profiler, "optimize_directions_with_initial_feasibility", forbidden)
    monkeypatch.setattr(profiler, "_construct", forbidden)
    monkeypatch.setattr(profiler, "_patterns", forbidden)
    parents = profiler.development_family("handover_heavy", 20)
    sequences = []
    for _ in range(2):
        sequence = []
        for ordinal in range(8):
            sampled = profiler.sample_method_independent_solution(
                parents,
                FAST,
                master_seed=20260929,
                family="handover_heavy",
                size=20,
                sample_ordinal=ordinal,
            )
            solution, directions, metadata = sampled
            assert canonicalize(
                solution.parents, solution.patterns, solution.routes, FAST
            ).canonical_json == solution.canonical_json
            assert tuple(len(row) for row in directions) == tuple(
                len(route.block_ids) for route in solution.routes
            )
            assert metadata["sampler_policy_id"] == "METHOD_INDEPENDENT_DIRECT_SAMPLER_V1"
            sequence.append((solution.canonical_json, directions, metadata))
        sequences.append(sequence)
    assert sequences[0] == sequences[1]


def test_method_independent_pattern_sampling_selects_kind_before_y_candidate() -> None:
    import profile_scheduler as profiler

    parent = ParentWeld("optional-many-y", (1.0, 5.8), (9.0, 10.0))

    class RecordingRng:
        def __init__(self):
            self.calls = []

        def randrange(self, stop):
            self.calls.append(("kind", stop))
            return 1

        def choice(self, values):
            self.calls.append(("candidate", len(values)))
            return values[-1]

    rng = RecordingRng()
    pattern = profiler._sample_formal_pattern(parent, FAST, rng)
    assert pattern.kind is SplitKind.Y_SPLIT
    assert rng.calls[0] == ("kind", 2)
    assert rng.calls[1][0] == "candidate"
    assert rng.calls[1][1] >= 2


def test_method_independent_identity_dedup_and_json_round_trip_are_stable() -> None:
    import profile_scheduler as profiler

    parents = profiler.development_family("load_skew", 20)
    solution, directions, metadata = profiler.sample_method_independent_solution(
        parents,
        FAST,
        master_seed=20260929,
        family="load_skew",
        size=20,
        sample_ordinal=3,
    )
    first = profiler._direct_state_exact_key(solution, directions, FAST)
    second = profiler._direct_state_exact_key(solution, directions, FAST)
    assert first == second
    baseline = reference_schedule_optimized(
        solution, FAST, orientations={robot: directions[robot] for robot in range(4)}
    )
    entry = profiler._direct_corpus_entry(
        solution, directions, FAST, metadata, baseline
    )
    import json

    restored = json.loads(json.dumps(entry, sort_keys=True, allow_nan=False))
    assert restored == entry
    assert profiler._solution_from_payload(restored["canonical_solution"]).canonical_json == solution.canonical_json


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        (ScheduleStatus.DEADLOCK, True),
        (ScheduleStatus.FEASIBLE, False),
        (ScheduleStatus.INFEASIBLE, False),
        (ScheduleStatus.NUMERIC_FAILURE, False),
    ),
)
def test_method_independent_corpus_accepts_only_baseline_deadlock(status, expected) -> None:
    import profile_scheduler as profiler

    assert profiler._baseline_status_is_collectible(status) is expected


def test_final_budget_selector_uses_smallest_90_percent_budget_and_runtime_gate() -> None:
    import profile_scheduler as profiler

    selection = profiler.select_final_deadlock_budget(
        {32: 89, 64: 90, 128: 100},
        {32: 1.0, 64: 7.9, 128: 7.0},
    )
    assert selection == 64
    assert selection == profiler.select_final_deadlock_budget(
        {32: 89, 64: 90, 128: 100},
        {32: 1.0, 64: 7.9, 128: 7.0},
    )
    assert profiler.select_final_deadlock_budget(
        {32: 90, 64: 95, 128: 100},
        {32: 8.1, 64: 8.0, 128: 7.0},
    ) == 64
    assert profiler.select_final_deadlock_budget(
        {32: 0, 64: 0, 128: 0},
        {32: 2.0, 64: 3.0, 128: 4.0},
    ) == 32
    assert profiler.select_final_deadlock_budget(
        {32: 89, 64: 90, 128: 100},
        {32: 8.1, 64: 8.1, 128: 8.1},
    ) is None
