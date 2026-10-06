from __future__ import annotations

import random
from dataclasses import replace

import pytest

from mrta_baselines.wag_vns import (
    AdaptedWAGConfig,
    _blocks_for,
    _initial_patterns,
    factorial_edge_combination,
    farthest_insertion,
    generate_wag_assignments,
    nearest_addition,
    toggle_optional_y,
    two_opt,
    wag_lns_candidates,
    wag_move_candidates,
    wag_swap_candidates,
)
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import ParentWeld, ScientificConfig, SplitKind
from mrta_reference.solution import block_map


def test_yr_wag_seed_rotation_uses_full_family_order_with_available_fallback():
    from mrta_baselines.wag_vns import transition_legal_pattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config = ScientificConfig()
    parents = (ParentWeld("both", (1.0, 5.9), (5.0, 7.0)),)
    solution = generate_wag_assignments(parents, _initial_patterns(parents, config, scope=FORMAL_SCOPE_V2),
                                       config, max_variants=1, scope=FORMAL_SCOPE_V2)[0]
    kinds = []
    for seed in (20261005, 20261006, 20261007):
        first = transition_legal_pattern(solution, config, random.Random(seed), FORMAL_SCOPE_V2,
                                        family_ordinal=seed, cyclic_family_priority=True)
        replay = transition_legal_pattern(solution, config, random.Random(seed), FORMAL_SCOPE_V2,
                                         family_ordinal=seed, cyclic_family_priority=True)
        assert first[0].canonical_hash == replay[0].canonical_hash
        kinds.append(first[1])
    assert SplitKind.Y_SPLIT in kinds and SplitKind.X_SPLIT in kinds


@pytest.mark.parametrize("v2", (False, True))
def test_yr_wag_per_iteration_pattern_first_and_v1_old_trigger(monkeypatch, v2):
    from mrta_baselines import wag_vns as wag
    from mrta_baselines.common import CommonBaselineEvaluator
    from mrta_reference.scope import FORMAL_SCOPE_V1_1, FORMAL_SCOPE_V2
    scope = FORMAL_SCOPE_V2 if v2 else FORMAL_SCOPE_V1_1
    config = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)
    parents = (ParentWeld("both", (1.0, 5.9), (5.0, 7.0)),)
    solution = generate_wag_assignments(parents, _initial_patterns(parents, config, scope=scope if v2 else None),
                                       config, max_variants=1, scope=scope if v2 else None)[0]
    seed_candidate = CommonBaselineEvaluator(parents, config, scope, time_limit=60).evaluate(solution, source="fixture")
    events = []
    original_evaluate = CommonBaselineEvaluator.evaluate
    def evaluate(self, assignment, *, source):
        events.append(source)
        return original_evaluate(self, assignment, source=source)
    monkeypatch.setattr(CommonBaselineEvaluator, "evaluate", evaluate)
    def heavy(*args, **kwargs):
        events.append("HEAVY")
    monkeypatch.setattr(wag, "_evaluate_routed_assignment", heavy)
    def candidates(solution, *args, **kwargs):
        events.append("OPERATOR")
        return (solution,)
    for name in ("wag_move_candidates", "wag_swap_candidates", "wag_lns_candidates"):
        monkeypatch.setattr(wag, name, candidates)
    original_toggle = wag.toggle_optional_y
    def toggle(*args):
        events.append("OLD_Y")
        return original_toggle(*args)
    monkeypatch.setattr(wag, "toggle_optional_y", toggle)
    result = wag.run_adapted_wag_vns(parents, config, replace(AdaptedWAGConfig(), imax=3), seed=20261006,
                                    time_limit=60, scope=scope, common_seed=seed_candidate,
                                    method_id=wag.METHOD_ID_V2 if v2 else wag.METHOD_ID)
    assert result.iterations == 3
    if v2:
        trace = result.accounting["pattern_transition_trace"]
        assert len(trace) == 3
        assert result.accounting["pattern_transition_proposals"] == 3
        assert events[0] == "WAG_PATTERN_TRANSITION_1_DIRECT"
        for iteration in (1, 2, 3):
            idx = events.index(f"WAG_PATTERN_TRANSITION_{iteration}_DIRECT")
            assert events[idx + 1] == "OPERATOR"
        assert "OLD_Y" not in events
        assert all(row["evaluation_started"] >= row["proposal_time"] for row in trace)
    else:
        assert not result.accounting["pattern_transition_trace"]
        assert events.count("OLD_Y") == 1
        assert not any("PATTERN_TRANSITION" in event for event in events)


def test_yr_wag_direct_transition_consumes_same_deadline_before_heavy(monkeypatch):
    from mrta_baselines import wag_vns as wag
    from mrta_baselines.common import CommonBaselineEvaluator
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)
    parents = (ParentWeld("one", (1.0, 2.0), (5.0, 2.0)),)
    solution = generate_wag_assignments(parents, _initial_patterns(parents, config, scope=FORMAL_SCOPE_V2),
                                       config, max_variants=1, scope=FORMAL_SCOPE_V2)[0]
    seed_candidate = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V2, time_limit=60).evaluate(solution, source="fixture")
    class CostlyTransitionEvaluator(CommonBaselineEvaluator):
        exhausted = False
        @property
        def expired(self):
            return self.exhausted or super().expired
        def evaluate(self, assignment, *, source):
            result = super().evaluate(assignment, source=source)
            if "TRANSITION" in source:
                self.exhausted = True
            return result
    monkeypatch.setattr(wag, "CommonBaselineEvaluator", CostlyTransitionEvaluator)
    def forbidden(*args, **kwargs):
        pytest.fail("heavy operator started after transition exhausted the shared deadline")
    for name in ("wag_move_candidates", "wag_swap_candidates", "wag_lns_candidates", "_evaluate_routed_assignment"):
        monkeypatch.setattr(wag, name, forbidden)
    result = wag.run_adapted_wag_vns(parents, config, AdaptedWAGConfig(), seed=20261006, time_limit=60,
                                    scope=FORMAL_SCOPE_V2, common_seed=seed_candidate, method_id=wag.METHOD_ID_V2)
    assert result.iterations == 1
    assert result.termination_reason == "TIME_LIMIT"
    assert len(result.accounting["pattern_transition_trace"]) == 1


def _parents():
    upper = tuple(
        ParentWeld(f"u{index}", (1.0 + index * 1.4, 9.0), (1.7 + index * 1.4, 9.0))
        for index in range(6)
    )
    lower = tuple(
        ParentWeld(f"l{index}", (1.3 + index * 1.4, 3.0), (2.0 + index * 1.4, 3.0))
        for index in range(6)
    )
    return upper + lower


def _assert_valid(solution, config):
    blocks = block_map(solution, config)
    ids = [item for route in solution.routes for item in route.block_ids]
    assert len(ids) == len(set(ids)) == len(blocks)
    assert all(
        robot_is_eligible(blocks[item], route.robot_id, config)
        for route in solution.routes
        for item in route.block_ids
    )


def test_wag_assignment_is_balanced_deterministic_multiple_and_eligible():
    config = ScientificConfig()
    parents = _parents()
    patterns = _initial_patterns(parents, config)
    first = generate_wag_assignments(parents, patterns, config, max_variants=10)
    second = generate_wag_assignments(parents, patterns, config, max_variants=10)
    assert len(first) > 1
    assert [item.canonical_json for item in first] == [item.canonical_json for item in second]
    for solution in first:
        _assert_valid(solution, config)


def test_nearest_farthest_two_opt_and_factorial_window_five():
    config = ScientificConfig()
    parents = _parents()[:5]
    patterns = _initial_patterns(parents, config)
    blocks = _blocks_for(parents, patterns, config)
    ids = tuple(blocks)
    nearest = nearest_addition(ids, 0, blocks, config)
    farthest = farthest_insertion(ids, blocks, config)
    assert set(nearest) == set(farthest) == set(ids)
    improved = two_opt(tuple(reversed(nearest)), blocks, config, consider_final_edge=True)
    assert set(improved) == set(ids)
    _, calls, windows = factorial_edge_combination(
        nearest,
        blocks,
        config,
        window_size=5,
        consider_final_edge=True,
        exact=True,
        max_windows=8,
        max_factorial_calls=30720,
    )
    assert calls >= 3840
    assert calls % 3840 == 0
    assert windows >= 1


def test_wag_move_swap_lns_same_rail_and_fixed_seed_reproducibility():
    config = ScientificConfig()
    parents = _parents()
    base = generate_wag_assignments(
        parents, _initial_patterns(parents, config), config, max_variants=1
    )[0]
    first_moves = wag_move_candidates(base, config, random.Random(7), n=10, heavy_robot=0)
    second_moves = wag_move_candidates(base, config, random.Random(7), n=10, heavy_robot=0)
    assert [item.canonical_json for item in first_moves] == [
        item.canonical_json for item in second_moves
    ]
    swaps = wag_swap_candidates(base, config, random.Random(8), n=10, heavy_robot=0)
    lns = wag_lns_candidates(base, config, random.Random(9), p=0.5, max_variants=3)
    assert first_moves and swaps and lns
    for candidate in first_moves + swaps + lns:
        _assert_valid(candidate, config)


def test_wag_optional_y_toggle_uses_only_legal_pattern():
    config = ScientificConfig()
    parents = _parents() + (ParentWeld("optional", (2.0, 5.9), (5.0, 7.0)),)
    base = generate_wag_assignments(
        parents, _initial_patterns(parents, config), config, max_variants=1
    )[0]
    toggled = toggle_optional_y(base, config)
    assert toggled is not None
    assert next(
        pattern for pattern in toggled.patterns if pattern.parent_id == "optional"
    ).kind is SplitKind.Y_SPLIT
    _assert_valid(toggled, config)


def test_common_model_bounded_defaults_preserve_paper_parameters():
    config = AdaptedWAGConfig()
    assert (config.n, config.p, config.imax, config.factorial_window) == (10, 0.5, 500, 5)
    assert config.factorial_mode == "COMMON_MODEL_BOUNDED"
