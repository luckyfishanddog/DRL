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


def test_spatial_native_follow_matches_original_on_motion_wait_and_blocking():
    import numpy as np
    from mrta_baselines import wag_vns as w
    config = ScientificConfig()
    rng = np.random.default_rng(861)
    for trial in range(200):
        points = rng.uniform((0., 0.), (20., 12.), (8, 2, 2))
        if trial % 3 == 0:
            points[:, :, 1] = 9.  # collinear motion and same-rail blocking
        processing = np.array([config.process_time(np.linalg.norm(p[1] - p[0])) for p in points])
        travel = np.empty((8, 8, 2, 2))
        for d in range(2):
            for e in range(2):
                delta = points[:, 1 - d, None, :] - points[None, :, e, :]
                travel[:, :, d, e] = np.sqrt(np.sum(delta * delta, axis=-1)) / config.empty_speed
        a, b = rng.permutation(4), rng.permutation(np.arange(4, 8))
        _, da = w._direction_array(a, travel, processing, trial % 2)
        _, db = w._direction_array(b, travel, processing, (trial // 2) % 2)
        na = w._numeric_templates(a, da, points, processing, config.empty_speed, config.t_pre, config.t_post)
        nb = w._numeric_templates(b, db, points, processing, config.empty_speed, config.t_pre, config.t_post)
        for fixed, following, rf, rr in ((na, nb, 0, 1), (nb, na, 1, 0)):
            ordinary, wait = w._numeric_follow(fixed, following, rf, rr, config.interference_dx,
                                                config.interference_dy, config.numeric_epsilon)
            for early in (False, True):
                skipped, other_wait = w._numeric_follow(fixed, following, rf, rr, config.interference_dx,
                                                        config.interference_dy, config.numeric_epsilon, True, True,
                                                        np.inf, True, early_reject=early)
                assert wait == other_wait
                np.testing.assert_array_equal(ordinary, skipped)


def test_routed_assignment_schedules_native_winner_once(monkeypatch):
    import math
    from mrta_baselines import wag_vns as w
    from mrta_baselines.common import CommonBaselineEvaluator
    from mrta_reference.model import Route, SplitPattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.solution import canonicalize
    config = ScientificConfig()
    parents = (ParentWeld('left', (1., 9.), (2., 9.)), ParentWeld('right', (8., 9.), (9., 9.)))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    solution = canonicalize(parents, patterns, (Route(0, ('left::whole',)), Route(1, ('right::whole',)),
                                               Route(2, ()), Route(3, ())), config, scope=FORMAL_SCOPE_V2)
    expected = w.priority_schedule(solution, config)
    routes = tuple(route.block_ids for route in solution.routes)
    monkeypatch.setattr(w, '_route_combinations', lambda *a, **k: (((0., routes),), 0))
    original, calls = w._priority_arrays, []
    def counted(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(w, '_priority_arrays', counted)
    evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V2, time_limit=math.inf,
                                        certify=False, candidate_evaluator=w.priority_schedule)
    w._evaluate_routed_assignment(solution, config, w.AdaptedWAGConfig(), evaluator, source='ONCE')
    assert len(calls) == 1
    assert evaluator.best.schedule == expected
    assert evaluator.accounting.reference_calls == 1


@pytest.mark.parametrize("with_splits", (False, True))
def test_lns_fixes_every_retained_task_owner_and_repairs_only_destroyed_tasks(with_splits):
    import math
    from mrta_baselines import wag_vns as w
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.model import Route
    from mrta_reference.solution import canonicalize
    config = ScientificConfig()
    parents = _parents()
    if with_splits:
        parents += (ParentWeld('x', (2., 9.), (14., 9.)), ParentWeld('y', (2., 5.9), (5., 7.)))
        catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
        patterns = tuple(next(p for p in catalog[parent.parent_id] if p.kind is
                         (SplitKind.X_SPLIT if parent.parent_id == 'x' else
                          SplitKind.Y_SPLIT if parent.parent_id == 'y' else SplitKind.WHOLE))
                         for parent in parents)
        routes = [tuple(f'u{i}::whole' for i in (0, 2, 4)) + ('x::0', 'y::1'),
                  tuple(f'u{i}::whole' for i in (1, 3, 5)) + ('x::1',),
                  tuple(f'l{i}::whole' for i in (0, 2, 4)) + ('y::0',),
                  tuple(f'l{i}::whole' for i in (1, 3, 5))]
        base = canonicalize(parents, patterns, tuple(Route(r, ids) for r, ids in enumerate(routes)),
                            config, scope=FORMAL_SCOPE_V2)
    else:
        base = generate_wag_assignments(parents, _initial_patterns(parents, config, scope=FORMAL_SCOPE_V2),
                                       config, max_variants=1, scope=FORMAL_SCOPE_V2)[0]
    old_blocks = block_map(base, config, scope=FORMAL_SCOPE_V2)
    class RecordingRandom(random.Random):
        def __init__(self, seed):
            super().__init__(seed)
            self.shuffled = []
        def shuffle(self, values):
            super().shuffle(values)
            self.shuffled.append(tuple(values))
    for seed in range(10):
        rng = RecordingRandom(seed)
        candidates = w.wag_lns_candidates(base, config, rng, p=.5, max_variants=None, scope=FORMAL_SCOPE_V2)
        retained = {name: robot for robot, ids in enumerate(rng.shuffled)
                    for name in ids[math.ceil(len(ids) * .5):]}
        assert retained and candidates
        for candidate in candidates:
            new_blocks = block_map(candidate, config, scope=FORMAL_SCOPE_V2)
            owner = {name: route.robot_id for route in candidate.routes for name in route.block_ids}
            assert set(owner) == set(new_blocks)
            for name, robot in retained.items():
                old = old_blocks[name]
                # Canonical sibling merging may change the task ID, but every
                # retained physical interval must remain on its original robot.
                covering = [b for b in new_blocks.values() if b.parent_id == old.parent_id
                            and b.u_start <= old.u_start and b.u_end >= old.u_end]
                assert covering and all(owner[b.block_id] == robot for b in covering)
            for pattern in candidate.patterns:
                if pattern.kind is SplitKind.X_SPLIT:
                    first = 0 if pattern.rail.value == 'UPPER' else 2
                    assert owner[pattern.parent_id + '::0'] == first
                    assert owner[pattern.parent_id + '::1'] == first + 1


def test_native_wag_completes_500_rounds_without_calling_b32_or_certifier(monkeypatch):
    import math
    from mrta_baselines import common, wag_vns as w
    config = ScientificConfig()
    parents = (ParentWeld('left', (1., 9.), (2., 9.)), ParentWeld('right', (8., 9.), (9., 9.)))
    def forbidden(*args, **kwargs):
        raise AssertionError('B32/certifier called inside native WAG search')
    monkeypatch.setattr(common.FormalReferenceEvaluator, '__call__', forbidden)
    monkeypatch.setattr(common, 'certify_schedule', forbidden)
    result = w.run_adapted_wag_vns_v2(parents, config, AdaptedWAGConfig(), seed=20261110,
                                     time_limit=math.inf, checkpoints=(), certify=False,
                                     candidate_evaluator=w.priority_schedule)
    assert result.iterations == 500
    assert result.termination_reason == 'ITERATION_LIMIT'
    assert result.certifier_calls == 0
    assert result.schedule.feasible


def test_lazy_time_window_preserves_epsilon_boundaries_zero_moves_and_waits():
    import numpy as np
    from mrta_baselines import wag_vns as w
    config = ScientificConfig()
    parents = tuple(ParentWeld(name, (x, 9.), (x + 1., 9.))
                    for name, x in (('a', 0.), ('b', 1.), ('c', .7), ('d', 1.7)))
    blocks = _blocks_for(parents, _initial_patterns(parents, config), config)
    _, _, points, travel, processing = w._tables_for(blocks, config)
    routes = (np.array([0, 1], np.int64), np.array([2, 3], np.int64))
    operations = []
    for route in routes:
        _, bits = w._direction_array(route, travel, processing, 0)
        operations.append(w._numeric_templates(route, bits, points, processing,
                                               config.empty_speed, config.t_pre, config.t_post))
    assert any(op[0] == 0. for op in operations[0])
    eps = config.numeric_epsilon
    for boundary in (0., 20., 30., operations[0][2, 6]):
        for offset in (-1.e-6, -1.e-8, -eps, 0., eps, 1.e-8, 1.e-6):
            fixed = operations[0].copy()
            fixed[:, 5:7] += boundary + offset
            for a, b, rf, rr in ((fixed, operations[1], 0, 1), (operations[1], fixed, 1, 0)):
                expected, wait = w._numeric_follow(a, b, rf, rr, config.interference_dx,
                                                   config.interference_dy, eps)
                for spatial in (False, True):
                    for early in (False, True):
                        actual, actual_wait = w._numeric_follow(a, b, rf, rr, config.interference_dx,
                                                                config.interference_dy, eps, spatial, True,
                                                                np.inf, True, early_reject=early)
                        assert actual_wait == wait
                        np.testing.assert_array_equal(actual, expected)


def test_lazy_projection_section_contains_every_original_interval():
    import numpy as np
    from mrta_baselines import wag_vns as w
    config = ScientificConfig()
    rng = np.random.default_rng(19073)
    for trial in range(2500):
        a, b = np.zeros(9), np.zeros(9)
        for op in (a, b):
            op[0] = 0. if trial % 11 == 0 else rng.uniform(.001, 300.)
            op[1:5] = rng.uniform(-2., 12., 4)
            op[5] = rng.uniform(0., 2000.)
            op[6] = op[5] + op[0]
        ready = rng.uniform(0., 2000.)
        for ra, rb in ((0, 1), (1, 0)):
            for order in (False, True):
                lo, hi = w._numeric_forbidden(a, b, ra, rb, ready, config.interference_dx,
                                               config.interference_dy, config.numeric_epsilon, order)
                specialized = w._numeric_forbidden_buffer(a, b, ra, rb, ready,
                    config.interference_dx, config.interference_dy, config.numeric_epsilon,
                    order, np.empty((10, 3)), True)
                assert specialized == (lo, hi)
                if lo > hi:
                    continue
                for start in (lo, hi, .5 * (lo + hi)):
                    assert w._forbidden_may_contain(a, b, ra, ready, start, config.interference_dx,
                                                     config.interference_dy, config.numeric_epsilon, order)


def test_scalar_direction_dp_preserves_exact_cost_and_lexicographic_ties():
    import itertools
    import numpy as np
    from mrta_baselines import wag_vns as w
    rng = np.random.default_rng(4781)
    for n in range(1, 9):
        route = np.arange(n, dtype=np.int64)
        processing = rng.integers(1, 8, n).astype(float)
        # Integer-valued arcs deliberately produce exact DP ties.
        travel = rng.integers(0, 4, (n, n, 2, 2)).astype(float)
        for first in range(2):
            choices = []
            for tail in itertools.product((0, 1), repeat=n - 1):
                bits = (first,) + tail
                value = processing[0]
                for k in range(1, n):
                    value = value + travel[k - 1, k, bits[k - 1], bits[k]] + processing[k]
                choices.append((value, bits))
            expected, bits = min(choices)
            actual, directions = w._direction_array(route, travel, processing, first)
            assert actual == expected and tuple(directions) == bits


def test_incremental_follow_prefix_matches_full_recomputation_after_suffix_changes():
    import numpy as np
    from mrta_baselines import wag_vns as w
    config, rng, reused = ScientificConfig(), np.random.default_rng(21965), 0
    for trial in range(200):
        points = rng.uniform((0., 8.), (12., 10.), (8, 2, 2))
        if trial % 2 == 0:
            points[:4, :, 0] *= .2
            points[4:, :, 0] = 8. + .2 * points[4:, :, 0]
        processing = np.array([config.process_time(np.linalg.norm(p[1] - p[0])) for p in points])
        travel = np.empty((8, 8, 2, 2))
        for d in range(2):
            for e in range(2):
                delta = points[:, 1 - d, None, :] - points[None, :, e, :]
                travel[:, :, d, e] = np.sqrt(np.sum(delta * delta, axis=-1)) / config.empty_speed
        templates = []
        for route in (np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)):
            _, bits = w._direction_array(route, travel, processing, trial % 2)
            templates.append(w._numeric_templates(route, bits, points, processing,
                                                  config.empty_speed, config.t_pre, config.t_post))
        for rf, rr in ((0, 1), (1, 0)):
            fixed, following = templates[rf], templates[rr]
            old_output, _ = w._numeric_follow(fixed, following, rf, rr, config.interference_dx,
                                               config.interference_dy, config.numeric_epsilon)
            if not len(old_output):
                continue
            new_fixed, new_following = fixed.copy(), following.copy()
            for modified in (new_fixed, new_following):
                index = int(rng.integers(1, len(modified)))
                delta = rng.choice((-1.e-8, 1.e-8, 1., 10.))
                if modified[index, 0] + delta < 0.:
                    delta = abs(delta)
                modified[index, 0] += delta
                modified[index, 6] += delta
                modified[index + 1:, 5:7] += delta
            reuse = (fixed, following, old_output)
            count, _, _, _ = w._follow_reusable_prefix(new_fixed, new_following, *reuse,
                                                       config.numeric_epsilon)
            reused += int(count > 0)
            expected, expected_wait = w._numeric_follow(new_fixed, new_following, rf, rr,
                config.interference_dx, config.interference_dy, config.numeric_epsilon)
            for early in (False, True):
                actual, actual_wait = w._numeric_follow(new_fixed, new_following, rf, rr,
                    config.interference_dx, config.interference_dy, config.numeric_epsilon,
                    True, True, np.inf, True, reuse, early_reject=early)
                assert actual_wait == expected_wait
                np.testing.assert_array_equal(actual, expected)
    assert reused > 50
