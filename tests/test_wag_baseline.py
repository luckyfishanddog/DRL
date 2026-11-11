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


def test_wag_boundary_includes_work_already_fixed_by_x_split():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config = ScientificConfig()
    parents = (ParentWeld('x', (2., 9.), (14., 9.)),
               ParentWeld('free', (1., 9.), (1.5, 9.)))
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    split = next(p for p in catalog['x']
                 if p.kind is SplitKind.X_SPLIT and p.point_id == 'BX_OUTER_LOWER')
    solution = generate_wag_assignments(parents, (split, catalog['free'][0]), config,
                                        max_variants=1, scope=FORMAL_SCOPE_V2)[0]
    # The right X child is longer. Putting the free weld on R1, as the old
    # empty-load calculation did, increases an existing imbalance.
    assert 'free::whole' in solution.routes[0].block_ids
    assert 'x::0' in solution.routes[0].block_ids
    assert 'x::1' in solution.routes[1].block_ids


def test_wag_shared_y_assignment_accounts_for_fixed_x_rail_load():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config = ScientificConfig()
    parents = (ParentWeld('x', (2., 3.), (14., 3.)),
               ParentWeld('shared', (17., 6.), (18., 6.)))
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    split = next(p for p in catalog['x'] if p.kind is SplitKind.X_SPLIT)
    solution = generate_wag_assignments(parents, (split, catalog['shared'][0]), config,
                                        max_variants=1, scope=FORMAL_SCOPE_V2)[0]
    # The lower rail already has the X children; it is not an empty rail tied
    # with the upper one when the shared-Y whole weld is assigned.
    owner = next(r.robot_id for r in solution.routes if 'shared::whole' in r.block_ids)
    assert owner in (0, 1)


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


def test_wag_defaults_preserve_paper_parameters_without_candidate_caps():
    config = AdaptedWAGConfig()
    assert (config.n, config.p, config.imax, config.factorial_window) == (10, 0.5, 500, 5)
    assert config.factorial_mode == "PAPER_EXACT"
    assert config.max_wag_variants is None
    assert config.max_route_combinations_per_wag is None


def test_incremental_splices_and_reversed_segments_match_full_direction_dp():
    import numpy as np
    from mrta_baselines import wag_vns as w
    rng = random.Random(812)
    config = ScientificConfig()
    parents = tuple(ParentWeld(str(i), (rng.uniform(1., 18.), rng.uniform(8., 11.)),
                               (rng.uniform(1., 18.), rng.uniform(8., 11.))) for i in range(8))
    blocks = _blocks_for(parents, _initial_patterns(parents, config), config)
    ids, _, _, travel, processing = w._tables_for(blocks, config)
    nodes = np.arange(len(ids), dtype=np.int64)
    prefix, suffix = w._route_prefix_suffix(nodes, travel, processing)
    matrices = w._interval_costs(nodes, travel, processing)
    for lo in range(len(ids) + 1):
        for hi in range(lo, len(ids) + 1):
            inserted = nodes[lo:hi][::-1].copy()
            expected = w._route_cost(ids[:lo] + tuple(ids[i] for i in inserted) + ids[hi:], blocks, config)
            assert w._splice_cost(nodes, prefix, suffix, lo, hi, inserted, travel, processing) == pytest.approx(expected, abs=1e-8)
            a, b, old = w._join_segment(0., 0., -1, nodes, matrices, 0, lo, False, travel)
            a, b, old = w._join_segment(a, b, old, nodes, matrices, lo, hi, True, travel)
            a, b, old = w._join_segment(a, b, old, nodes, matrices, hi, len(nodes), False, travel)
            assert min(a, b) == pytest.approx(expected, abs=1e-8)


def test_five_edge_factorial_matches_full_order_and_direction_enumeration():
    import itertools
    import math
    from mrta_baselines import wag_vns as w
    config = ScientificConfig()
    parents = tuple(ParentWeld(str(i), (x, 8. + i % 2), (x + .7, 9. + i % 2))
                    for i, x in enumerate((7., 1., 11., 3., 14.)))
    blocks = _blocks_for(parents, _initial_patterns(parents, config), config)
    best = math.inf
    for order in itertools.permutations(blocks):
        for bits in itertools.product((0, 1), repeat=5):
            value = sum(config.process_time(blocks[n].length) for n in order)
            value += sum(math.dist((blocks[a].end, blocks[a].start)[da],
                                   (blocks[b].start, blocks[b].end)[db]) / config.empty_speed
                         for a, b, da, db in zip(order, order[1:], bits, bits[1:]))
            best = min(best, value)
    result, calls, _ = factorial_edge_combination(tuple(blocks), blocks, config, window_size=5,
                                                 consider_final_edge=True, exact=True,
                                                 max_windows=1, max_factorial_calls=1)
    assert w._route_cost(result, blocks, config) == pytest.approx(best, abs=1e-8)
    assert calls >= 3840


def test_all_81_combinations_can_win_beyond_the_old_top_three(monkeypatch):
    from types import SimpleNamespace
    from mrta_baselines import wag_vns as w
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_baselines.common import BaselineAccounting
    config = ScientificConfig()
    parents = _parents()
    assignment = generate_wag_assignments(parents, _initial_patterns(parents, config), config, max_variants=1)[0]
    routes = tuple(r.block_ids for r in assignment.routes)
    monkeypatch.setattr(w, '_route_combinations', lambda *args, **kwargs: ([(0., routes)] * 81, 0))
    class Evaluator:
        deadline, expired, best, scope = float('inf'), False, None, FORMAL_SCOPE_V2
        accounting = BaselineAccounting()
        count = 0
        def evaluate(self, solution, *, source):
            self.count += 1
            self.best = SimpleNamespace(metrics=SimpleNamespace(cmax=200. - self.count))
    evaluator = Evaluator()
    w._evaluate_routed_assignment(assignment, config, AdaptedWAGConfig(), evaluator, source='all')
    assert evaluator.count == 81
    assert evaluator.best.metrics.cmax == 119.


def test_combination_lower_bound_accounts_for_canonical_sibling_merge(monkeypatch):
    from mrta_baselines import wag_vns as w
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.model import CanonicalSolution
    from mrta_reference.solution import canonicalize
    config = ScientificConfig()
    parents = (ParentWeld('p', (1., 5.9), (5., 7.)), ParentWeld('q', (7., 9.), (8., 9.)))
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    patterns = (next(p for p in catalog['p'] if p.kind is SplitKind.Y_SPLIT), catalog['q'][0])
    solution = canonicalize(parents, patterns,
                            {0: ('p::0', 'q::whole', 'p::1'), 1: (), 2: (), 3: ()},
                            config, scope=FORMAL_SCOPE_V2)
    def adjacent(ids, robot, *args, **kwargs):
        route = ('p::0', 'p::1', 'q::whole') if robot == 0 else ()
        return (route, route, route), 0
    monkeypatch.setattr(w, 'improved_route_set', adjacent)
    combinations, _ = w._route_combinations(solution, config, AdaptedWAGConfig(),
                                            deadline=float('inf'), scope=FORMAL_SCOPE_V2)
    assert len(combinations) == 81
    bound, normalized = combinations[0]
    assert isinstance(normalized, CanonicalSolution)
    assert next(p for p in normalized.patterns if p.parent_id == 'p').kind is SplitKind.WHOLE
    actual = block_map(normalized, config)
    assert bound == pytest.approx(w._route_cost(normalized.routes[0].block_ids, actual, config), abs=1e-8)


def test_compiled_priority_follow_matches_existing_ess_including_wait():
    import math
    import numpy as np
    from mrta_baselines import wag_vns as w
    from mrta_reference.model import RobotRoute
    from mrta_reference.scheduler import build_operation_templates, earliest_safe_start
    config = ScientificConfig()
    parents = tuple(ParentWeld(f'p{i}', (x, 9.), (x + 1., 9.))
                    for i, x in enumerate((1., 7., 3., 10., 6., 14.)))
    blocks = _blocks_for(parents, _initial_patterns(parents, config), config)
    ids, _, points, travel, processing = w._tables_for(blocks, config)
    a, b = np.arange(3, dtype=np.int64), np.arange(3, 6, dtype=np.int64)
    _, da = w._direction_array(a, travel, processing, 0)
    _, db = w._direction_array(b, travel, processing, 0)
    na = w._numeric_templates(a, da, points, processing, config.empty_speed, config.t_pre, config.t_post)
    nb = w._numeric_templates(b, db, points, processing, config.empty_speed, config.t_pre, config.t_post)
    result, wait = w._numeric_follow(na, nb, 0, 1, config.interference_dx,
                                     config.interference_dy, config.numeric_epsilon)
    fixed = build_operation_templates(RobotRoute(0, tuple(blocks[ids[i]] for i in a), tuple(da)), config)
    following = build_operation_templates(RobotRoute(1, tuple(blocks[ids[i]] for i in b), tuple(db)), config)
    ready = waited = 0.
    for operation in following:
        start, _, _ = earliest_safe_start(operation, ready, fixed, config)
        assert math.isfinite(start)
        waited += start - ready
        ready = start + operation.duration
    assert wait > 0.
    assert wait == pytest.approx(waited, abs=1e-7)
    assert result[-1, 6] == pytest.approx(ready, abs=1e-7)
