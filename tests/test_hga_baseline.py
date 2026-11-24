from __future__ import annotations

import random

import pytest

from mrta_baselines.hga import (
    _Individual,
    alpha_nearest,
    apply_hga_neighborhood,
    hga_surrogate,
    initialize_hga_region_seed,
    initialize_hga_solution,
    manage_population,
    mutate_optional_y,
    mutate_legal_pattern,
    normalized_route_edge_distance,
    route_based_crossover,
    AdaptedHGAConfig,
)
from mrta_baselines.wag_vns import transition_legal_pattern
from mrta_reference.geometry import (
    build_legal_pattern_catalog,
    pattern_catalog_hash,
    robot_is_eligible,
)
from mrta_reference.model import (
    FORMAL_SCOPE_V2,
    MoveType,
    ParentWeld,
    Route,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.solution import block_map, canonicalize
from mrta_search.neighborhood import make_raw_candidate

# SAEAX shares only the existing numeric geometry/scorer with HGA. Keep its
# directly related regressions here instead of adding another test file.


def _parents():
    return tuple(
        ParentWeld(chr(97 + index), (1.0 + index * 1.5, 9.0), (1.7 + index * 1.5, 9.0))
        for index in range(8)
    )


def _fixture():
    config = ScientificConfig()
    parents = _parents()
    patterns = tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents)
    solution = canonicalize(
        parents,
        patterns,
        (
            Route(0, ("a::whole", "b::whole", "c::whole", "d::whole")),
            Route(1, ("e::whole", "f::whole", "g::whole", "h::whole")),
            Route(2, ()),
            Route(3, ()),
        ),
        config,
    )
    return config, parents, solution


def _assert_valid(solution, config):
    blocks = block_map(solution, config)
    ids = [item for route in solution.routes for item in route.block_ids]
    assert len(ids) == len(set(ids)) == len(blocks)
    assert set(ids) == set(blocks)
    assert all(
        robot_is_eligible(blocks[item], route.robot_id, config)
        for route in solution.routes
        for item in route.block_ids
    )


def test_v2_three_method_pattern_access_uses_one_catalog_hash():
    config = ScientificConfig()
    parents = (
        ParentWeld("upper", (1.0, 8.0), (5.0, 8.0)),
        ParentWeld("lower", (6.0, 4.0), (10.0, 4.0)),
        ParentWeld("handover", (2.0, 5.0), (3.0, 7.0)),
    )
    catalog = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    expected_hash = pattern_catalog_hash(parents, config, FORMAL_SCOPE_V2)
    assert expected_hash == pattern_catalog_hash(
        tuple(reversed(parents)), config, FORMAL_SCOPE_V2
    )
    initial = initialize_hga_region_seed(
        parents, config, scope=FORMAL_SCOPE_V2
    )
    hga_candidate, hga_kind = mutate_legal_pattern(
        initial,
        config,
        random.Random(7),
        FORMAL_SCOPE_V2,
        family_ordinal=1,
    )
    wag_candidate, wag_kind = transition_legal_pattern(
        initial,
        config,
        random.Random(7),
        FORMAL_SCOPE_V2,
        family_ordinal=1,
    )
    alns = make_raw_candidate(
        initial,
        config,
        MoveType.SPLIT_ACTIVATE,
        seed=7,
        ordinal=1,
        enable_x_split=True,
        scope=FORMAL_SCOPE_V2,
    )
    assert hga_kind is wag_kind is SplitKind.X_SPLIT
    assert wag_candidate is not None
    assert alns.candidate is not None
    assert alns.candidate.split_pattern is not None
    assert alns.candidate.split_pattern.kind is SplitKind.X_SPLIT
    for solution in (hga_candidate, wag_candidate):
        for pattern in solution.patterns:
            assert pattern in catalog[pattern.parent_id]


def test_hga_population_initialization_is_fixed_seed_deterministic_and_eligible():
    config = ScientificConfig()
    parents = _parents()
    first = initialize_hga_solution(parents, config, random.Random(17))
    second = initialize_hga_solution(parents, config, random.Random(17))
    assert first.canonical_json == second.canonical_json
    _assert_valid(first, config)

    region_seed = initialize_hga_region_seed(parents, config)
    _assert_valid(region_seed, config)
    assert region_seed.canonical_json == initialize_hga_region_seed(
        parents, config
    ).canonical_json


def test_route_based_crossover_preserves_every_block_once_and_is_deterministic():
    config, parents, first = _fixture()
    second = canonicalize(
        parents,
        first.patterns,
        (
            Route(0, ("h::whole", "g::whole", "f::whole", "e::whole")),
            Route(1, ("d::whole", "c::whole", "b::whole", "a::whole")),
            Route(2, ()),
            Route(3, ()),
        ),
        config,
    )
    child1 = route_based_crossover(first, second, config, random.Random(99))
    child2 = route_based_crossover(first, second, config, random.Random(99))
    assert child1.canonical_json == child2.canonical_json
    _assert_valid(child1, config)


def test_all_six_hga_neighborhoods_exist_and_preserve_canonical_coverage():
    config, _, solution = _fixture()
    cases = {
        1: ("a::whole", "e::whole"),
        2: ("a::whole", "e::whole"),
        3: ("a::whole", "e::whole"),
        4: ("a::whole", "c::whole"),
        5: ("a::whole", "e::whole"),
        6: ("a::whole", "e::whole"),
    }
    for move, pair in cases.items():
        candidate = apply_hga_neighborhood(solution, move, *pair, config)
        assert candidate is not None, f"M{move} was not constructed"
        _assert_valid(candidate, config)


def test_hga_two_opt_reconnections_follow_the_paper_cut_edges():
    config, _, solution = _fixture()
    m4 = apply_hga_neighborhood(solution, 4, 'a::whole', 'c::whole', config)
    assert m4.routes[0].block_ids == ('a::whole', 'c::whole', 'b::whole', 'd::whole')
    m5 = apply_hga_neighborhood(solution, 5, 'b::whole', 'f::whole', config)
    assert m5.routes[0].block_ids == ('a::whole', 'b::whole', 'f::whole', 'e::whole')
    assert m5.routes[1].block_ids == ('d::whole', 'c::whole', 'g::whole', 'h::whole')
    m6 = apply_hga_neighborhood(solution, 6, 'b::whole', 'f::whole', config)
    assert m6.routes[0].block_ids == ('a::whole', 'b::whole', 'g::whole', 'h::whole')
    assert m6.routes[1].block_ids == ('e::whole', 'f::whole', 'c::whole', 'd::whole')


def test_crossover_missing_insertion_minimizes_cycle_time_before_travel_delta():
    import numpy as np
    from mrta_baselines import hga as h
    config = ScientificConfig()
    parents = (ParentWeld('a', (1., 9.), (2., 9.)), ParentWeld('b', (4., 9.), (5., 9.)),
               ParentWeld('c', (2., 9.), (4., 9.)))
    workspace = h._HGAWorkspace(parents, config, None)
    blocks = h._blocks_for(parents, h.initial_patterns(parents, config), config)
    routes = [['a::whole', 'b::whole'], [], [], []]
    # Inserting c into R0 replaces its 2 m empty arc, making delta smaller
    # there, but inserting it into empty R1 avoids increasing the bottleneck.
    assert h._best_insertion('c::whole', routes, blocks, config, shortest_route_only=False) == (1, 0)
    lookup = {name: i for i, name in enumerate(workspace.ids)}
    arrays = [np.array([lookup[name] for name in route], np.int64) for route in routes]
    assert h._insert_arrays(lookup['c::whole'], arrays, workspace.eligible, workspace.travel,
                             workspace.processing, False) == (1, 0)


def test_alpha_nearness_cap_diversity_population_management_and_optional_y():
    config, _, solution = _fixture()
    assert len(alpha_nearest(solution, "a::whole", 20, config)) == 7
    assert len(alpha_nearest(solution, "a::whole", 3, config)) == 3
    reversed_solution = canonicalize(
        solution.parents,
        solution.patterns,
        tuple(Route(route.robot_id, tuple(reversed(route.block_ids))) for route in solution.routes),
        config,
    )
    distance = normalized_route_edge_distance(solution, reversed_solution)
    assert 0.0 < distance <= 1.0
    assert distance == normalized_route_edge_distance(reversed_solution, solution)
    population = [
        _Individual(solution, hga_surrogate(solution, config)),
        _Individual(reversed_solution, hga_surrogate(reversed_solution, config)),
    ]
    assert len(manage_population(population, 1)) == 1

    optional = ParentWeld("optional", (2.0, 5.9), (5.0, 7.0))
    optional_solution = initialize_hga_solution((optional,), config, random.Random(3))
    mutated = mutate_optional_y(optional_solution, config, random.Random(3))
    assert mutated.patterns[0].kind is SplitKind.Y_SPLIT
    _assert_valid(mutated, config)


def test_hga_defaults_preserve_paper_budget_and_disable_vnd_caps():
    config = AdaptedHGAConfig()
    assert (config.mu, config.lambda_size, config.alpha, config.max_iterations) == (
        20,
        10,
        20,
        200000,
    )
    assert config.initialization_reference_limit == config.mu
    assert config.vnd_candidate_cap is None
    assert config.vnd_max_passes is None


def test_m1_m6_segment_costs_match_materialized_full_route_costs():
    import numpy as np
    import pytest
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _interval_costs, _array_route_cost, _direction_array
    config, _, solution = _fixture()
    workspace = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = workspace.encode(solution)
    summaries = [_interval_costs(route, workspace.travel, workspace.processing) for route in routes]
    nodes = np.full((4, 5), -1, np.int64)
    lengths = np.array([len(route) for route in routes], np.int64)
    matrices = np.zeros((4, 5, 5, 2, 2))
    for r in range(4):
        nodes[r, :lengths[r]] = routes[r]
        matrices[r, :lengths[r], :lengths[r]] = summaries[r]
    invalid = np.zeros((4, 4, 5), np.int64)
    for move in range(1, 7):
        ru, iu, rv, iv = (0, 0, 0, 2) if move == 4 else (0, 0, 1, 0)
        a, b = h._move_segments(routes, move, ru, iu, rv, iv)
        cu, cv = h._raw_move_costs(nodes, lengths, matrices, invalid, workspace.parent_index,
                                   workspace.travel, move, ru, iu, rv, iv)
        for target, segments in ((ru, a), (rv, b)):
            if move == 4 and segments is b:
                continue
            materialized = h._materialize_segments(segments, routes)
            cost = _array_route_cost(materialized, workspace.travel, workspace.processing)
            assert (cu if target == ru else cv) == pytest.approx(cost, abs=1e-8)
            head, head_costs = h._segment_head_costs(segments, nodes, matrices, invalid,
                                                    workspace.parent_index, workspace.travel, target)
            assert head == (materialized[0] if len(materialized) else -1)
            for first in range(2):
                expected, _ = _direction_array(materialized, workspace.travel, workspace.processing, first)
                assert head_costs[first] == pytest.approx(expected, abs=1e-8)
    # Invalid first segments must stay invalid when later segments are joined.
    invalid[0, 0, :] = np.arange(5)
    cu, _ = h._raw_move_costs(nodes, lengths, matrices, invalid, workspace.parent_index,
                              workspace.travel, 2, 0, 2, 1, 0)
    assert cu == float('inf')


def test_unbounded_vnd_stops_only_after_all_six_neighborhoods_have_no_improvement():
    from mrta_baselines import hga as h
    from mrta_baselines.common import BaselineAccounting
    config, _, solution = _fixture()
    accounting = BaselineAccounting()
    result = h.hga_vnd(solution, config, alpha=20, accounting=accounting)
    _assert_valid(result, config)
    assert accounting.vnd_pass_cap_hits == accounting.vnd_candidate_cap_hits == 0
    from mrta_baselines.wag_vns import priority_schedule
    score = priority_schedule(result, config).cmax
    workspace = h._HGAWorkspace(result.parents, config, None, patterns=result.patterns)
    for move in range(1, 7):
        for route in result.routes:
            for u in route.block_ids:
                for v in h.alpha_nearest(result, u, 20, config):
                    candidate = h.apply_hga_neighborhood(result, move, u, v, config)
                    if candidate is not None and h._legal_first_arrays(workspace.encode(candidate), workspace.points, workspace.parameters):
                        native = priority_schedule(candidate, config)
                        assert not native.feasible or native.cmax >= score - 1e-8


def test_hga_first_task_repair_keeps_coverage_and_fixes_same_rail_order():
    from mrta_baselines.hga import repair_first_tasks
    from mrta_search.direction import optimize_directions_with_initial_feasibility, DirectionStatus
    config, parents, solution = _fixture()
    bad = canonicalize(parents, solution.patterns, (Route(0, ('d::whole', 'a::whole')),
                        Route(1, ('b::whole', 'c::whole', 'e::whole', 'f::whole', 'g::whole', 'h::whole')),
                        Route(2, ()), Route(3, ())), config)
    assert optimize_directions_with_initial_feasibility(bad, config).status is not DirectionStatus.FEASIBLE
    repaired = repair_first_tasks(bad, config)
    _assert_valid(repaired, config)
    assert optimize_directions_with_initial_feasibility(repaired, config).status is DirectionStatus.FEASIBLE


def test_native_hga_retries_failed_child_without_spending_iteration_or_calling_b32(monkeypatch):
    from dataclasses import replace
    import math
    from mrta_baselines import hga as h
    from mrta_baselines import common
    from mrta_baselines.wag_vns import priority_schedule
    parents = (ParentWeld('a', (1., 9.), (2., 9.)), ParentWeld('b', (8., 9.), (9., 9.)))
    original = h._HGAWorkspace.repair
    calls = []
    def repair(self, *args, **kwargs):
        calls.append(None)
        if len(calls) == 3:  # first child after the two initialized individuals
            raise ValueError('deliberately rejected construction')
        return original(self, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('B32/certifier called inside native HGA')
    monkeypatch.setattr(h._HGAWorkspace, 'repair', repair)
    monkeypatch.setattr(common.FormalReferenceEvaluator, '__call__', forbidden)
    monkeypatch.setattr(common, 'certify_schedule', forbidden)
    result = h.run_adapted_hga_v2(parents, ScientificConfig(),
                                  replace(AdaptedHGAConfig(), mu=2, lambda_size=1, max_iterations=3),
                                  seed=20261110, time_limit=math.inf, checkpoints=(), certify=False,
                                  candidate_evaluator=priority_schedule)
    assert result.iterations == 3
    assert result.termination_reason == 'ITERATION_LIMIT'
    assert result.accounting['construction_rejected'] == 1
    assert result.certifier_calls == 0
    assert result.accounting['vnd_cache_hits'] > 0
    assert result.accounting['vnd_cache_hits'] + result.accounting['vnd_unique_calls'] == 5
    assert result.accounting['vnd_candidate_operations'] == result.accounting['cheap_candidate_operations']
    _assert_valid(result.solution, ScientificConfig())


def test_incremental_native_pair_reuse_matches_full_native_for_all_six_moves():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    original = w.encode(solution)
    empty_bits = [np.empty(0, np.int64) for _ in range(4)]
    empty_ops = [np.empty((0, 9)) for _ in range(4)]
    _, bits, ops, pairs = h._native_update(original, w.travel, w.processing, w.points, w.parameters,
                                          empty_bits, empty_ops, np.zeros(2), np.ones(4, np.bool_), np.inf)
    for move in range(1, 7):
        for iu in range(len(original[0])):
            rv = 0 if move == 4 else 1
            for iv in range(len(original[rv])):
                if move == 3 and (iu + 1 == len(original[0]) or iv + 1 == len(original[rv])):
                    continue
                a, b = h._move_segments(original, move, 0, iu, rv, iv)
                routes = list(original)
                routes[0] = h._materialize_segments(a, original)
                if move != 4:
                    routes[rv] = h._materialize_segments(b, original)
                dirty = np.array([True, rv == 1, False, False])
                value, ds, os, _ = h._native_update(routes, w.travel, w.processing, w.points, w.parameters,
                                                    bits, ops, pairs, dirty, np.inf)
                expected, ed, eo, _ = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)
                assert value == expected
                if np.isfinite(value):
                    for actual, oracle in zip(ds, ed):
                        np.testing.assert_array_equal(actual, oracle)
                    for actual, oracle in zip(os, eo):
                        np.testing.assert_array_equal(actual, oracle)


def test_m4_improves_coordinated_cmax_on_a_nonlongest_route():
    import numpy as np
    import pytest
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays, _array_route_cost
    coordinates = ((3.1896431826810465, .21917399373940183), (3.705434752413466, .6320241177990604),
                   (.9724609631777217, .4744404435755521), (3.1658989239446154, .6575792865855388),
                   (5.09087957129091, .8221933942533106), (8.540593049487441, .5244224216873925),
                   (1.2204224481040233, .5334787799689157), (2.3953114349999427, .8331953740528919))
    config = ScientificConfig()
    parents = tuple(ParentWeld(str(i), (x, 9.), (x + length, 9.)) for i, (x, length) in enumerate(coordinates))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    solution = canonicalize(parents, patterns, (Route(0, tuple(f'{i}::whole' for i in range(4))),
                            Route(1, tuple(f'{i}::whole' for i in range(4, 8))), Route(2, ()), Route(3, ())), config)
    w = h._HGAWorkspace(parents, config, None, patterns=patterns)
    routes = w.encode(solution)
    before = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)[0]
    assert _array_route_cost(routes[0], w.travel, w.processing) < _array_route_cost(routes[1], w.travel, w.processing)
    # Isolate intra-route moves with robot eligibility in this kernel fixture.
    # Formal runs continue to use all actual qualifications and all six moves.
    eligible = np.zeros_like(w.eligible)
    for r, route in enumerate(routes):
        eligible[route, r] = True
    result, score, operations, accepted, _, _ = h._vnd_arrays(
        routes, w.travel, w.processing, w.points, eligible, w.parent_index, w.nearest,
        w.parameters, 20, 1, 0, True, w.vnd_scratch)
    assert accepted == 1 and operations > 0
    assert not np.array_equal(result[0], routes[0])
    np.testing.assert_array_equal(result[1], routes[1])
    assert score < before
    assert score == pytest.approx(_priority_arrays(result, w.travel, w.processing, w.points, w.parameters)[0])
    snapshots = [route.copy() for route in routes]
    reused = h._vnd_arrays(routes, w.travel, w.processing, w.points, eligible, w.parent_index, w.nearest,
                          w.parameters, 20, 1, 0, True, w.vnd_scratch, np.zeros(1, np.int64))
    assert reused[1:] == (score, operations, accepted, True, False)
    for actual, expected in zip(reused[0], result):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(routes, snapshots):
        np.testing.assert_array_equal(actual, expected)
    # The old max-independent-route objective cannot see this improvement.
    assert max(_array_route_cost(r, w.travel, w.processing) for r in result) == pytest.approx(
        max(_array_route_cost(r, w.travel, w.processing) for r in routes))


def test_native_incremental_reuse_with_four_nonempty_routes_and_cross_rail_checks():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config = ScientificConfig()
    upper = _parents()
    lower = tuple(ParentWeld('lower_' + p.parent_id, (p.start[0], 3.), (p.end[0], 3.)) for p in upper)
    parents = upper + lower
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    solution = canonicalize(parents, patterns, tuple(Route(r, tuple(p.parent_id + '::whole' for p in group))
                            for r, group in enumerate((upper[:4], upper[4:], lower[:4], lower[4:]))), config)
    w = h._HGAWorkspace(parents, config, None, patterns=patterns)
    base = w.encode(solution)
    _, bits, ops, pairs = h._native_update(base, w.travel, w.processing, w.points, w.parameters,
        [np.empty(0, np.int64) for _ in range(4)], [np.empty((0, 9)) for _ in range(4)],
        np.zeros(2), np.ones(4, np.bool_), np.inf)
    rng, feasible = np.random.default_rng(72), 0
    original_bits, original_ops, original_pairs = [a.copy() for a in bits], [a.copy() for a in ops], pairs.copy()
    for trial in range(40):
        routes = [route.copy() for route in base]
        dirty = np.zeros(4, np.bool_)
        for r in ((trial % 4,) if trial % 2 else (0, 2)):
            routes[r] = rng.permutation(routes[r])
            dirty[r] = True
        value, ds, os, _ = h._native_update(routes, w.travel, w.processing, w.points, w.parameters,
                                           bits, ops, pairs, dirty, np.inf)
        oracle, ed, eo, _ = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)
        assert value == oracle
        if np.isfinite(value):
            feasible += 1
            for actual, expected in zip(ds, ed):
                np.testing.assert_array_equal(actual, expected)
            for actual, expected in zip(os, eo):
                np.testing.assert_array_equal(actual, expected)
            # A cutoff may suppress only a non-improving result. Both priorities
            # remain available, including when just one reaches the cutoff.
            for limit in (oracle - 1.e-5, oracle + 1.e-5):
                bounded, bd, bo, _ = h._native_update(routes, w.travel, w.processing, w.points, w.parameters,
                                                      bits, ops, pairs, dirty, limit)
                if oracle < limit - 1.e-8:
                    assert bounded == oracle
                    for actual, expected in zip(bo, eo):
                        np.testing.assert_array_equal(actual, expected)
                else:
                    assert bounded >= limit - 1.e-8
    assert feasible > 0
    for actual, expected in zip(bits, original_bits):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(ops, original_ops):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pairs, original_pairs)
    # The same numeric geometry with overlapping rails must still be rejected
    # exactly as the full native scorer, rather than trusting pair feasibility.
    overlapping = w.points.copy()
    overlapping[base[2], :, 1] = 9.
    overlapping[base[3], :, 1] = 9.
    value, *_ = h._native_update(base, w.travel, w.processing, overlapping, w.parameters,
        bits, ops, pairs, np.ones(4, np.bool_), np.inf)
    assert value == _priority_arrays(base, w.travel, w.processing, overlapping, w.parameters)[0] == np.inf
    # Initialization must retain both pair states even if the first pair is
    # blocked; a later candidate may change that first pair alone.
    blocked = [r.copy() for r in base]
    blocked[0], blocked[1] = blocked[1], blocked[0]
    value, ds, os, ps = h._native_update(blocked, w.travel, w.processing, w.points, w.parameters,
        bits, ops, pairs, np.ones(4, np.bool_), np.inf)
    assert not np.isfinite(value)
    assert ps[1] == pairs[1] and len(ds[2]) and len(ds[3])


def test_native_head_bound_never_discards_an_improving_neighborhood():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    base = w.encode(solution)
    nodes, lengths, summaries, invalid, prefix, robots, indexes = w.vnd_scratch
    heads, costs = np.full(4, -1, np.int64), np.zeros((4, 2))
    for r in range(4):
        h._refresh_vnd_route(r, base[r], w.travel, w.processing, w.eligible, nodes, lengths,
                             summaries, invalid, prefix, robots, indexes)
        if len(base[r]):
            heads[r] = base[r][0]
            costs[r] = np.min(summaries[r, 0, len(base[r]) - 1], axis=1)
    score, directions, ops, _ = _priority_arrays(base, w.travel, w.processing, w.points, w.parameters)
    checked, rejected = 0, 0
    for move in range(1, 7):
        for iu in range(3):
            for iv in range(3):
                ru, rv = 0, 0 if move == 4 else 1
                if move == 4 and iu == iv:
                    continue
                a, b = h._move_segments(base, move, ru, iu, rv, iv)
                routes, ch, cc = list(base), heads.copy(), costs.copy()
                changed = np.zeros(4, np.bool_)
                for r, segments in ((ru, a),) if move == 4 else ((ru, a), (rv, b)):
                    routes[r] = h._materialize_segments(segments, base)
                    ch[r], cc[r] = h._segment_head_costs(segments, nodes, summaries, invalid,
                                                         w.parent_index, w.travel, r)
                    changed[r] = True
                oracle = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)[0]
                for limit in (score, oracle - 1.e-5, oracle + 1.e-5) if np.isfinite(oracle) else (score,):
                    allowed = h._pair_head_bound(0, ch, cc, w.points, w.parameters, ops, directions,
                                                  changed, limit)
                    checked += 1
                    if not allowed:
                        rejected += 1
                        assert oracle >= limit - 1.e-8
                    if oracle < limit - 1.e-8:
                        assert allowed
    assert checked > 60 and rejected > 0


def test_vnd_cache_remains_bounded_when_each_result_introduces_two_keys():
    from mrta_baselines.hga import _remember_vnd
    cache = {}
    for i in range(5000):
        individual = object()
        key, terminal = ('input', i), ('terminal', i)
        _remember_vnd(cache, key, terminal, individual)
        assert len(cache) <= 4096
        assert cache[key] is cache[terminal] is individual
    assert ('input', 0) not in cache
    individual = object()
    _remember_vnd(cache, terminal, terminal, individual)
    assert len(cache) == 4096 and cache[terminal] is individual


def test_symmetric_move_reuse_preserves_full_vnd_and_counts_visited_proposals():
    import numpy as np
    from mrta_baselines import hga as h
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    args = (routes, w.travel, w.processing, w.points, w.eligible, w.parent_index, w.nearest,
            w.parameters, 20, 0, 0, True, w.vnd_scratch)
    full = h._vnd_arrays(*args)
    hits = np.zeros(1, np.int64)
    reused = h._vnd_arrays(*args, hits)
    for actual, expected in zip(reused[0], full[0]):
        np.testing.assert_array_equal(actual, expected)
    assert reused[1:] == full[1:]
    assert hits[0] > 0
    for move in (2, 3, 4, 6):
        for iu in range(3):
            for iv in range(3):
                ru, rv = 0, 0 if move == 4 else 1
                if move == 4 and iu == iv:
                    continue
                candidates = []
                for r, i, s, j in ((ru, iu, rv, iv), (rv, iv, ru, iu)):
                    a, b = h._move_segments(routes, move, r, i, s, j)
                    candidate = list(routes)
                    candidate[r] = h._materialize_segments(a, routes)
                    if move != 4:
                        candidate[s] = h._materialize_segments(b, routes)
                    candidates.append(candidate)
                for actual, expected in zip(*candidates):
                    np.testing.assert_array_equal(actual, expected)


def test_interval_head_selection_and_near_tie_fallback_match_full_native():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _direction_array, _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    base = w.encode(solution)
    _, bits, ops, pairs = h._native_update(base, w.travel, w.processing, w.points, w.parameters,
        [np.empty(0, np.int64) for _ in range(4)], [np.empty((0, 9)) for _ in range(4)],
        np.zeros(2), np.ones(4, np.bool_), np.inf)
    rng, unique, ties = np.random.default_rng(4869), 0, 0
    for trial in range(40):
        routes = [rng.permutation(route) for route in base]
        costs = np.array([[_direction_array(route, w.travel, w.processing, first)[0]
                           for first in range(2)] for route in routes])
        unique += int(h._unique_pair_heads(routes, 0, costs, w.points, w.parameters)[0] >= 0)
        # Force the conservative fallback even if the estimates are perturbed
        # or tied; it must use the original direction DP and its exact tie rule.
        if trial % 2:
            costs[:] = 0.
            ties += int(h._unique_pair_heads(routes, 0, costs, w.points, w.parameters)[0] < 0)
        actual, ad, ao, _ = h._native_update(routes, w.travel, w.processing, w.points, w.parameters,
            bits, ops, pairs, np.ones(4, np.bool_), np.inf, None, costs)
        expected, ed, eo, _ = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)
        assert actual == expected
        if np.isfinite(actual):
            for a, b in zip(ad, ed):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(ao, eo):
                np.testing.assert_array_equal(a, b)
    assert unique > 0 and ties > 0


def test_oriented_six_moves_preserve_or_complement_exactly_the_affected_genes():
    import numpy as np
    from mrta_baselines import hga as h
    routes = [np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64),
              np.empty(0, np.int64), np.empty(0, np.int64)]
    bits = [np.array([0, 1, 0, 1], np.int64), np.array([1, 0, 1, 0], np.int64),
            np.empty(0, np.int64), np.empty(0, np.int64)]
    expected = {
        1: (((0, 0), (2, 0), (3, 1)), ((4, 1), (5, 0), (1, 1), (6, 1), (7, 0))),
        2: (((0, 0), (5, 0), (2, 0), (3, 1)), ((4, 1), (1, 1), (6, 1), (7, 0))),
        3: (((0, 0), (5, 0), (6, 1), (3, 1)), ((4, 1), (1, 1), (2, 0), (7, 0))),
        4: (((0, 0), (2, 1), (1, 0), (3, 1)), ((4, 1), (5, 0), (6, 1), (7, 0))),
        5: (((0, 0), (1, 1), (5, 1), (4, 0)), ((3, 0), (2, 1), (6, 1), (7, 0))),
        6: (((0, 0), (1, 1), (6, 1), (7, 0)), ((4, 1), (5, 0), (2, 0), (3, 1))),
    }
    for move in range(1, 7):
        ru, iu, rv, iv = (0, 0, 0, 2) if move == 4 else (0, 1, 1, 1)
        a, b = h._move_segments(routes, move, ru, iu, rv, iv)
        candidate, directions = list(routes), list(bits)
        for r, segments in ((ru, a),) if move == 4 else ((ru, a), (rv, b)):
            candidate[r] = h._materialize_segments(segments, routes)
            directions[r] = h._materialize_directions(segments, bits)
        assert h._oriented_identity(candidate, directions)[:2] == expected[move]
    a, _ = h._move_segments(routes, 4, 0, 0, 0, 1)
    np.testing.assert_array_equal(h._materialize_segments(a, routes), routes[0])
    np.testing.assert_array_equal(h._materialize_directions(a, bits), [0, 0, 0, 1])


def test_oriented_interval_costs_equal_materialized_fixed_direction_costs():
    import numpy as np
    import pytest
    from mrta_baselines import hga as h
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in routes]
    for r in range(4):
        h._refresh_oriented_route(r, routes[r], bits[r], w.travel, w.processing, w.eligible, w.oriented_scratch)
    for move in range(1, 7):
        for iu in range(3):
            for iv in range(3):
                ru, rv = 0, 0 if move == 4 else 1
                if move == 4 and iu == iv:
                    continue
                a, b = h._move_segments(routes, move, ru, iu, rv, iv)
                for r, segments in ((ru, a),) if move == 4 else ((ru, a), (rv, b)):
                    route, directions = h._materialize_segments(segments, routes), h._materialize_directions(segments, bits)
                    expected = h._oriented_route_cost(route, directions, w.travel, w.processing)
                    actual = h._oriented_segment_cost(segments, r, w.oriented_scratch, w.travel, w.parent_index)
                    assert actual == pytest.approx(expected, abs=1.e-8)


def test_fixed_gene_native_updates_match_full_priority_for_all_six_moves():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.zeros(len(r), np.int64) for r in routes]
    empty = [np.empty((0, 9)) for _ in range(4)]
    score, ops, pairs, nominal, follows = h._fixed_native_update(routes, bits, w.points, w.processing,
        w.parameters, empty, np.zeros(2), empty, empty, np.ones(4, np.bool_), np.inf)
    assert np.isfinite(score)
    for move in range(1, 7):
        for iu in range(3):
            for iv in range(3):
                ru, rv = 0, 0 if move == 4 else 1
                if move == 4 and iu == iv:
                    continue
                a, b = h._move_segments(routes, move, ru, iu, rv, iv)
                candidate, directions, dirty = list(routes), list(bits), np.zeros(4, np.bool_)
                for r, segments in ((ru, a),) if move == 4 else ((ru, a), (rv, b)):
                    candidate[r], directions[r] = h._materialize_segments(segments, routes), h._materialize_directions(segments, bits)
                    dirty[r] = True
                expected, ed, eo, _ = _priority_arrays(candidate, w.travel, w.processing, w.points, w.parameters, directions)
                for limit in (np.inf, score, expected - 1.e-5, expected + 1.e-5) if np.isfinite(expected) else (np.inf, score):
                    actual, actual_ops, _, _, _ = h._fixed_native_update(candidate, directions, w.points, w.processing,
                        w.parameters, ops, pairs, nominal, follows, dirty, limit)
                    if expected < limit - 1.e-8:
                        assert actual == expected
                        for aa, bb in zip(actual_ops, eo):
                            np.testing.assert_array_equal(aa, bb)
                    else:
                        assert not np.isfinite(actual) or actual >= limit - 1.e-8


def test_oriented_candidate_arrays_preserve_directed_order_and_symmetric_first_encounter():
    import numpy as np
    from mrta_baselines import hga as h
    routes = ([8, 1, 5], [4, 0, 9], [7, 3], [6, 2])
    all_ids = np.array(sum((list(r) for r in routes), []), np.int64)
    lengths = np.array([len(r) for r in routes], np.int64)
    robot, position = np.empty(10, np.int64), np.empty(10, np.int64)
    for r, route in enumerate(routes):
        for i, u in enumerate(route):
            robot[u], position[u] = r, i
    near = np.array([[8, 4, 3], [4, 5, 7], [9, 6, 3], [7, 8, 5], [1, 0, 7],
                     [8, 3, 2], [2, 7, 4], [3, 2, 4], [1, 5, -1], [8, 4, 2]], np.int64)
    candidates = np.empty((6, len(all_ids) * near.shape[1], 2), np.int64)
    counts = np.full(6, -1, np.int64)
    index = {u: i for i, u in enumerate(all_ids)}
    # 9 -> 8 exists, 8 -> 9 does not, and 9 occurs later in all_ids.
    assert index[9] > index[8] and 8 in near[9] and 9 not in near[8]
    for longest in range(4):
        h._collect_oriented_candidates(all_ids, near, robot, position, lengths, longest, candidates, counts)
        for move in range(1, 7):
            expected = []
            for u in all_ids:
                ru, iu = robot[u], position[u]
                if move == 1 and ru != longest:
                    continue
                for v in near[u]:
                    if v < 0:
                        break
                    rv, iv = robot[v], position[v]
                    if move == 1 and ru == rv:
                        continue
                    if move in (2, 3) and (ru == rv or longest not in (ru, rv)):
                        continue
                    if move == 3 and (iu + 1 >= lengths[ru] or iv + 1 >= lengths[rv]):
                        continue
                    if move == 4 and (ru != rv or iu == iv):
                        continue
                    if move in (5, 6) and ru == rv:
                        continue
                    expected.append((int(u), int(v)))
            actual = [tuple(pair) for pair in candidates[move - 1, :counts[move - 1]]]
            assert actual == expected
            if move in (2, 3, 4, 6):
                def first_encounters(pairs):
                    seen, first = set(), []
                    for u, v in pairs:
                        key = frozenset((u, v))
                        if key not in seen:
                            first.append((u, v))
                            seen.add(key)
                    return first
                assert first_encounters(actual) == first_encounters(expected)
            if longest == 0 and move == 2:
                assert (9, 8) in actual


@pytest.mark.parametrize('route_objective', (False, True))
def test_oriented_vnd_exhausts_all_six_fixed_gene_neighborhoods(route_objective):
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in routes]
    result = h._vnd_oriented_arrays(routes, bits, w.travel, w.processing, w.points, w.eligible, w.parent_index,
                                    w.nearest, w.parameters, 20, 0, 0, w.oriented_scratch, np.zeros(1, np.int64),
                                    route_objective=route_objective)
    current, directions, score, operations, _, pass_hit, candidate_hit, _ = result
    assert np.isfinite(score) and operations > 0 and not pass_hit and not candidate_hit
    expected = (max(h._oriented_route_cost(r, d, w.travel, w.processing) for r, d in zip(current, directions))
                if route_objective else
                _priority_arrays(current, w.travel, w.processing, w.points, w.parameters, directions)[0])
    assert score == pytest.approx(expected, abs=1.e-8)
    longest = int(np.argmax([h._oriented_route_cost(r, d, w.travel, w.processing) for r, d in zip(current, directions)]))
    for move in range(1, 7):
        for ru in range(4):
            for rv in range(4):
                if move == 1 and (ru == rv or ru != longest):
                    continue
                if move in (2, 3) and (ru == rv or longest not in (ru, rv)):
                    continue
                if move == 4 and ru != rv or move in (5, 6) and ru == rv:
                    continue
                for iu in range(len(current[ru])):
                    for iv in range(len(current[rv])):
                        if move == 4 and iu == iv or move == 3 and (iu + 1 >= len(current[ru]) or iv + 1 >= len(current[rv])):
                            continue
                        a, b = h._move_segments(current, move, ru, iu, rv, iv)
                        candidate, db = list(current), list(directions)
                        for r, segments in ((ru, a),) if move == 4 else ((ru, a), (rv, b)):
                            candidate[r], db[r] = h._materialize_segments(segments, current), h._materialize_directions(segments, directions)
                        if any(not w.eligible[n, r] for r, route in enumerate(candidate) for n in route):
                            continue
                        value = (max(h._oriented_route_cost(r, d, w.travel, w.processing) for r, d in zip(candidate, db))
                                 if route_objective else
                                 _priority_arrays(candidate, w.travel, w.processing, w.points, w.parameters, db)[0])
                        assert not np.isfinite(value) or value >= score - 1.e-8


def test_route_cycle_vnd_never_calls_native_coordination(monkeypatch):
    import numpy as np
    from mrta_baselines import hga as h
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in routes]
    def forbidden(*args, **kwargs):
        raise AssertionError('native coordination called inside route-cycle VND')
    for name in ('_priority_arrays', '_fixed_native_update', '_fixed_native_pair', '_numeric_follow'):
        monkeypatch.setattr(h, name, forbidden)
    hits = np.zeros(4, np.int64)
    result = h._vnd_oriented_arrays.py_func(routes, bits, w.travel, w.processing, w.points, w.eligible,
        w.parent_index, w.nearest, w.parameters, 20, 0, 0, w.oriented_scratch, hits, route_objective=True)
    current, directions, score, operations, _, pass_hit, candidate_hit, _ = result
    assert operations > 0 and hits[3] == 0 and not pass_hit and not candidate_hit
    assert score == pytest.approx(max(h._oriented_route_cost(r, d, w.travel, w.processing)
                                     for r, d in zip(current, directions)), abs=1.e-8)
    _assert_valid(w.solution(current, w.initial_selected), config)
    assert all(len(r) == len(d) for r, d in zip(current, directions))


def test_route_cycle_children_use_one_native_evaluation_and_reuse_best_timeline(monkeypatch):
    from dataclasses import replace
    from mrta_baselines import common, hga as h
    config = ScientificConfig()
    parents = (ParentWeld('a', (1., 9.), (2., 9.)), ParentWeld('b', (8., 9.), (9., 9.)))
    original, calls = h._priority_arrays, []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result[0])
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('B32/certifier called inside route-cycle HGA')
    monkeypatch.setattr(h, '_priority_arrays', observed)
    monkeypatch.setattr(common.FormalReferenceEvaluator, '__call__', forbidden)
    monkeypatch.setattr(common, 'certify_schedule', forbidden)
    settings = replace(AdaptedHGAConfig(), mu=3, lambda_size=2, max_iterations=4, vnd_objective='ROUTE_CYCLE')
    result = h.run_adapted_hga_v2(parents, config, settings, seed=731, time_limit=float('inf'), checkpoints=())
    assert result.iterations == 4 and result.termination_reason == 'ITERATION_LIMIT'
    initial_proposals = min(settings.mu, settings.initialization_reference_limit)
    assert len(calls) == result.reference_calls == settings.mu + result.iterations + initial_proposals
    assert result.accounting['vnd_unique_calls'] + result.accounting['vnd_cache_hits'] == settings.mu + result.iterations
    assert result.accounting['vnd_cache_hits'] > 0
    assert result.accounting['vnd_native_candidate_evaluations'] == 0 and result.certifier_calls == 0
    best = result.timely_best
    expected = h.priority_schedule(best.solution, config, best.directions)
    assert best.schedule.cmax == min(calls) == expected.cmax
    assert best.schedule.operations == expected.operations


def _oriented_wait_fixture(two_rails=False):
    from mrta_baselines import hga as h
    coordinates = ((3.1896431826810465, .21917399373940183), (3.705434752413466, .6320241177990604),
                   (.9724609631777217, .4744404435755521), (3.1658989239446154, .6575792865855388),
                   (5.09087957129091, .8221933942533106), (8.540593049487441, .5244224216873925),
                   (1.2204224481040233, .5334787799689157), (2.3953114349999427, .8331953740528919))
    rails = (('u', 9.), ('l', 3.)) if two_rails else (('u', 9.),)
    parents = tuple(ParentWeld(prefix + str(i), (x, y), (x + length, y))
                    for prefix, y in rails for i, (x, length) in enumerate(coordinates))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    config = ScientificConfig()
    groups = tuple(Route(r, tuple((('u' if r < 2 else 'l') + str(i) + '::whole')
                                  for i in range(4 * (r % 2), 4 * (r % 2 + 1))))
                   if r < 2 or two_rails else Route(r, ()) for r in range(4))
    solution = canonicalize(parents, patterns, groups, config)
    return h._HGAWorkspace(parents, config, None, patterns=patterns), solution


def test_oriented_vnd_repairs_two_native_rejected_pairs_in_successive_moves():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    w, solution = _oriented_wait_fixture(True)
    routes = w.encode(solution)
    directions = [np.array([0, 0, 1, 0], np.int64) for _ in range(4)]
    empty = [np.empty((0, 9)) for _ in range(4)]
    score, _, pairs, _, _ = h._fixed_native_update(routes, directions, w.points, w.processing, w.parameters,
        empty, np.zeros(2), empty, empty, np.ones(4, np.bool_), np.inf)
    assert not np.isfinite(score) and not np.isfinite(pairs).any()
    result = h._vnd_oriented_arrays(routes, directions, w.travel, w.processing, w.points, w.eligible,
        w.parent_index, w.nearest, w.parameters, 20, 0, 0, w.oriented_scratch, np.zeros(3, np.int64))
    current, bits, value, _, accepted, pass_hit, candidate_hit, rejected = result
    assert np.isfinite(value) and rejected == 0 and accepted >= 2
    assert not pass_hit and not candidate_hit
    assert value == _priority_arrays(current, w.travel, w.processing, w.points, w.parameters, bits)[0]


@pytest.mark.parametrize('rejected',[False,True])
def test_priority_progress_is_passive_and_matches_existing_coordinated_pair_rule(rejected):
    import math
    import numpy as np
    from mrta_baselines import hga as h
    w,solution=_oriented_wait_fixture(rejected)
    routes=w.encode(solution)
    directions=[np.array([0,0,1,0][:len(route)],np.int64) for route in routes]
    bits=tuple(tuple(int(b) for b in route) for route in directions)
    original=h.priority_schedule(solution,w.config,directions=bits)
    progress={}
    observed=h.priority_schedule(solution,w.config,directions=bits,progress=progress)
    assert observed.canonical_json()==original.canonical_json() and observed.directions==bits
    empty=[np.empty((0,9)) for _ in range(4)]
    _,_,pairs,_,_=h._fixed_native_update(routes,directions,w.points,w.processing,w.parameters,
        empty,np.zeros(2),empty,empty,np.ones(4,np.bool_),np.inf)
    assert progress['rejected_pairs']==sum(not math.isfinite(value) for value in pairs)


@pytest.mark.parametrize('route_objective', (False, True))
def test_oriented_prefix_bounds_match_complete_native_search_with_waits(route_objective):
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    w, solution = _oriented_wait_fixture()
    routes = w.encode(solution)
    _, bits, original_ops, _ = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)
    assert any(op[8] == 4. for group in original_ops for op in group)
    args = (routes, bits, w.travel, w.processing, w.points, w.eligible, w.parent_index, w.nearest,
            w.parameters, 20, 0, 0, w.oriented_scratch)
    full = h._vnd_oriented_arrays(*args, np.zeros(3, np.int64), False, route_objective=route_objective)
    hits = np.zeros(3, np.int64)
    bounded = h._vnd_oriented_arrays(*args, hits, True, route_objective=route_objective)
    for actual, expected in zip(bounded[0] + bounded[1], full[0] + full[1]):
        np.testing.assert_array_equal(actual, expected)
    assert bounded[2:] == full[2:]
    if route_objective:
        return

    # This small VND need not encounter a reusable failed priority. Exercise a
    # real later gene edit against the independent, complete follower scorer.
    from mrta_baselines.wag_vns import _numeric_follow, _numeric_templates
    empty = [np.empty((0, 9)) for _ in range(4)]
    failures = np.full((4, 2), np.nan)
    score, _, _, nominal, follows = h._fixed_native_update(
        routes, bits, w.points, w.processing, w.parameters, empty, np.zeros(2),
        empty, empty, np.ones(4, np.bool_), np.inf, failure_out=failures)
    assert failures[0, 0] == 10  # reverse priority fails before the final task
    edited_bits = [d.copy() for d in bits]
    edited_bits[0][-1] ^= 1
    costs = np.array([h._oriented_route_cost(r, d, w.travel, w.processing)
                      for r, d in zip(routes, edited_bits)])
    progress = [h._fixed_follow_progress(n, f) for n, f in zip(nominal, follows)]
    assert h._fixed_prefix_reject(1, 0, np.array([3, 4, 0, 0]),
        np.array([True, False, False, False]), nominal, progress, costs, score,
        w.parameters[-1], failures)
    candidate = _numeric_templates(routes[0], edited_bits[0], w.points, w.processing,
                                  *w.parameters[:3])
    _, wait = _numeric_follow(nominal[1], candidate, 1, 0, *w.parameters[3:])
    assert not np.isfinite(wait)


def test_oriented_repair_inherits_genes_and_preserves_all_legal_pattern_families():
    import numpy as np
    from mrta_baselines import hga as h
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in routes]
    repaired, directions = w.repair(routes, w.initial_selected, directions=bits)
    for actual, expected in zip(repaired + directions, routes + bits):
        np.testing.assert_array_equal(actual, expected)
    inherited, inherited_bits = list(routes), list(bits)
    inherited[0], inherited_bits[0] = routes[1], bits[1]
    repaired, directions = w.repair(inherited, w.initial_selected, 0, directions=inherited_bits)
    donor = dict(zip(routes[1], bits[1]))
    for route, db in zip(repaired, directions):
        for i in range(1, len(route)):
            if route[i] in donor:
                assert db[i] == donor[route[i]]
    _assert_valid(w.solution(repaired, w.initial_selected), config)
    parents = (ParentWeld('upper', (1., 8.), (5., 8.)), ParentWeld('lower', (6., 4.), (10., 4.)),
               ParentWeld('handover', (2., 5.), (3., 7.)))
    w = h._HGAWorkspace(parents, config, FORMAL_SCOPE_V2)
    base_solution = h.initialize_hga_region_seed(parents, config, scope=FORMAL_SCOPE_V2)
    base = w.encode(base_solution)
    bits = [h._greedy_route_directions(route, w.travel) for route in base]
    families = set()
    for parent, options in enumerate(w.options):
        for option in options:
            selected = w.initial_selected.copy()
            selected[parent] = option
            routes, db = w.repair(base, selected, directions=bits)
            _assert_valid(w.solution(routes, selected), config)
            assert all(len(route) == len(d) and np.all((d == 0) | (d == 1)) for route, d in zip(routes, db))
            assert all(w.eligible[n, r] for r, route in enumerate(routes) for n in route)
            families.add(w.patterns[option].kind)
    assert families == {SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT}


def test_explicit_hga_keeps_existing_finite_budget_callers_without_internal_b32(monkeypatch):
    from dataclasses import replace
    from mrta_baselines import common, hga as h
    config = ScientificConfig()
    parents = (ParentWeld('a', (1., 9.), (2., 9.)), ParentWeld('b', (8., 9.), (9., 9.)))
    def forbidden(*args, **kwargs):
        raise AssertionError('B32/certifier called inside explicit HGA')
    monkeypatch.setattr(common.FormalReferenceEvaluator, '__call__', forbidden)
    monkeypatch.setattr(common, 'certify_schedule', forbidden)
    settings = replace(AdaptedHGAConfig(), max_iterations=2)
    result = h.run_adapted_hga_v2(parents, config, settings, seed=731, time_limit=60., checkpoints=(60.,))
    assert result.iterations == 2 and result.termination_reason == 'ITERATION_LIMIT'
    assert result.timely_best is not None and result.checkpoints[60.] is not None
    assert result.certifier_calls == 0
    expired = h.run_adapted_hga_v2(parents, config, settings, seed=731, time_limit=1.e-12, checkpoints=())
    assert expired.iterations == 0 and expired.termination_reason == 'TIME_LIMIT'
    assert expired.timely_best is None and expired.certifier_calls == 0


def _saeax_fixture():
    import numpy as np
    from mrta_search import saeax as s
    config,parents,solution = _fixture()
    parents = tuple(ParentWeld(p.parent_id,p.start,(p.start[0]+1.2,9.)) for p in parents)
    solution = canonicalize(parents,solution.patterns,solution.routes,config,scope=FORMAL_SCOPE_V2)
    w = s._HGAWorkspace(parents,config,FORMAL_SCOPE_V2)
    def individual(sequences):
        lookup = {name:i for i,name in enumerate(w.ids)}
        routes = [np.array([lookup[n] for n in names],np.int64) for names in sequences]
        directions = [np.zeros(len(r),np.int64) for r in routes]
        costs,loads = s._route_statistics(routes,directions,w.travel,w.processing)
        edges = s._numeric_edges([2*r+d for r,d in zip(routes,directions)],2*len(w.ids))
        return s.Individual(routes,directions,w.initial_selected.copy(),float(max(costs)),costs,loads,
            s.complete_identity(routes,directions,w.initial_selected),edges)
    a = individual([r.block_ids for r in solution.routes])
    b = individual([('a::whole','c::whole','b::whole','d::whole'),
                    ('e::whole','g::whole','f::whole','h::whole'),(),()])
    return s,w,a,b


def test_saeax_directed_successor_predecessor_and_inactive_modes():
    s,w,a,b = _saeax_fixture()
    table,owners = s.enhanced_adjacency(a,b,a.selected,w)
    u,v = int(a.routes[0][0]),int(a.routes[0][1])
    assert any(row[0:3] == (v,0,0) for row in table[u,0])
    assert any(row[0:3] == (u,1,0) for row in table[v,1])
    assert owners[u] == [(0,0),(0,1)]
    selected = a.selected.copy()
    p = next(i for i,options in enumerate(w.options) if len(options)>1)
    selected[p] = next(o for o in w.options[p] if o != selected[p])
    inactive = set(w.pattern_nodes[a.selected[p]])-{-1}
    table,_ = s.enhanced_adjacency(a,b,selected,w)
    assert not any(node in inactive or row[0] in inactive
        for (node,_),rows in table.items() for row in rows)
    assert s.complete_identity(a.routes,a.directions,selected) != a.identity


def test_saeax_crossover_recombines_both_parents_without_hga_route_repair(monkeypatch):
    s,w,a,b = _saeax_fixture()
    def forbidden(*args,**kwargs):
        raise AssertionError('SAEAX crossover delegated to HGA repair')
    monkeypatch.setattr(w,'repair',forbidden)
    settings = s.SAEAXConfig(pattern_mutation_probability=0.,relocate_probability=0.,swap_probability=0.)
    def arcs(item):
        return {(int(r[k-1]),int(d[k-1]),int(r[k]),int(d[k]))
                for r,d in zip(item.routes,item.directions) for k in range(1,len(r))}
    aa,bb = arcs(a),arcs(b)
    mixed = False
    identities = set()
    for seed in range(24):
        routes,bits,selected,inherited,novel = s.adjacency_crossover(a,b,w,random.Random(seed),settings)
        s.validate_individual(w,routes,bits,selected)
        child_arcs = {(int(r[k-1]),int(d[k-1]),int(r[k]),int(d[k]))
                      for r,d in zip(routes,bits) for k in range(1,len(r))}
        mixed |= bool(child_arcs.intersection(aa-bb) and child_arcs.intersection(bb-aa))
        assert inherited+novel == sum(max(0,len(r)-1) for r in routes)
        identities.add(s.complete_identity(routes,bits,selected))
        solution = w.solution(routes,selected)
        _assert_valid(solution,w.config)
    assert mixed
    assert len(identities)>2


@pytest.mark.parametrize('owner0',[0,1])
@pytest.mark.parametrize('owner1',[2,3])
def test_saeax_y_children_have_independent_owners_and_x_is_fixed(owner0,owner1):
    import numpy as np
    from mrta_search import saeax as s
    parents = (ParentWeld('y',(3.,8.),(3.,4.)),ParentWeld('x',(1.,9.),(10.,9.)))
    w = s._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    selected = w.initial_selected.copy()
    x = next(i for i,p in enumerate(w.parents) if p.parent_id=='x')
    selected[x] = next(p for p in w.options[x] if w.patterns[p].kind is SplitKind.X_SPLIT)
    routes = [[] for _ in range(4)]
    for p in selected:
        for child,n in enumerate(w.pattern_nodes[p]):
            if n<0:
                continue
            owner = (owner0,owner1)[child] if w.patterns[p].kind is SplitKind.Y_SPLIT else child
            routes[owner].append(int(n))
    routes = [np.array(r,np.int64) for r in routes]
    bits = [np.zeros(len(r),np.int64) for r in routes]
    s.validate_individual(w,routes,bits,selected)
    _assert_valid(w.solution(routes,selected),w.config)
    bad = [r.copy() for r in routes]
    node = next(int(n) for n in routes[0] if w.patterns[w.node_pattern[n]].kind is SplitKind.X_SPLIT)
    bad[0] = bad[0][bad[0]!=node]
    bad[1] = np.append(bad[1],node)
    with pytest.raises(ValueError,match='qualification|ownership'):
        s.validate_individual(w,bad,[np.zeros(len(r),np.int64) for r in bad],selected)


def test_saeax_rejects_coverage_and_direction_corruption():
    import numpy as np
    s,w,a,_ = _saeax_fixture()
    bad = [r.copy() for r in a.routes]
    bad[0][0] = bad[0][1]
    with pytest.raises(ValueError,match='duplicate'):
        s.validate_individual(w,bad,a.directions,a.selected)
    bits = [d.astype(float) for d in a.directions]
    bits[0][0] = .5
    with pytest.raises(ValueError,match='direction'):
        s.validate_individual(w,a.routes,bits,a.selected)
    with pytest.raises(ValueError,match='direction'):
        s.validate_individual(w,a.routes,[np.empty(0,np.int64) for _ in range(4)],a.selected)


def test_saeax_survival_keeps_global_elite_despite_diversity():
    from dataclasses import replace
    s,_,a,b = _saeax_fixture()
    elite = replace(a,native_cmax=1.)
    assert s.select_population([b,elite],1)[0] is elite


@pytest.mark.parametrize('maximum,minimum,stagnation,expected,reason',[
    (8,5,3,5,'STAGNATION'),(3,1,10,3,'MAX_GENERATIONS')])
def test_saeax_natural_stop_accounting_and_b32_selection_are_separate(
        monkeypatch,maximum,minimum,stagnation,expected,reason):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    s,w,_,_ = _saeax_fixture()
    calls,progress = [],[]
    monkeypatch.setattr(s,'_priority_arrays',lambda *args:(100.,None,None,None))
    def b32(solution,config,*,scope,orientations):
        assert progress[-1]['generation']==expected
        calls.append((solution,orientations))
        directions = tuple(orientations[r] for r in range(4))
        return (ScheduleResult(ScheduleStatus.DEADLOCK,directions=directions) if len(calls)==1 else
                ScheduleResult(ScheduleStatus.FEASIBLE,cmax=1000.-100*len(calls),directions=directions))
    monkeypatch.setattr(s,'reference_schedule_formal',b32)
    settings = s.SAEAXConfig(population_size=4,offspring_per_generation=2,
        min_generations=minimum,max_generations=maximum,stagnation_generations=stagnation)
    parents = tuple(ParentWeld(f'p{i:02d}',(1.+i*1.1,9.),(1.7+i*1.1,9.)) for i in range(16))
    result = s.run_saeax(parents,settings=settings,seed=987,progress_callback=progress.append)
    assert result.generations==expected and result.valid_children==2*expected
    assert result.termination_reason==reason
    assert result.accounting['native_calls']==result.accounting['initialization_evaluations']+2*expected
    assert result.accounting['native_best_cmax_s']==100.
    assert result.status=='FEASIBLE' and result.cmax==700. and result.native_cmax==100.
    assert result.accounting['b32_calls']==len(calls)==3
    assert len({(x.canonical_json,tuple(d.items())) for x,d in calls})==3
    assert result.runtime>=result.accounting['search_s']+result.accounting['b32_s']


def test_saeax_repeated_children_stop_with_bounded_retries_and_no_fake_cmax(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    s,w,_,_ = _saeax_fixture()
    monkeypatch.setattr(s,'_priority_arrays',lambda *args:(100.,None,None,None))
    monkeypatch.setattr(s,'adjacency_crossover',lambda a,*args:
        (a.routes,a.directions,a.selected,0,0))
    monkeypatch.setattr(s,'reference_schedule_formal',lambda *args,**kwargs:ScheduleResult(ScheduleStatus.DEADLOCK))
    result = s.run_saeax(w.parents,settings=s.SAEAXConfig(population_size=2,
        offspring_per_generation=2,max_generations=80,min_generations=80,retries_per_child=2))
    assert result.generations==0 and result.valid_children==0
    assert result.termination_reason=='NOVEL_CHILD_RETRIES_EXHAUSTED'
    assert result.accounting['construction_attempts']==4
    assert result.accounting['native_calls']==1 and result.accounting['b32_calls']==1
    assert result.cmax is None and result.native_cmax==100. and result.status=='DEADLOCK'
    assert result.solution is not None and result.accounting['wait_is_partial'] is True


def test_saeax_defaults_and_generation_bounds():
    from mrta_search.saeax import SAEAXConfig
    config = SAEAXConfig()
    assert (config.population_size,config.offspring_per_generation,config.max_generations,
            config.min_generations,config.stagnation_generations)==(12,4,400,80,120)
    with pytest.raises(ValueError):
        SAEAXConfig(max_generations=10)
    with pytest.raises(ValueError):
        SAEAXConfig(retries_per_child=0)


def test_vnd_teacher_trace_preserves_search_and_observed_actions():
    import numpy as np
    from mrta_baselines import hga as h
    config,parents,solution = _fixture()
    solution = canonicalize(parents,solution.patterns,
        {0:('a::whole','c::whole','b::whole','d::whole'),
         1:solution.routes[1].block_ids,2:(),3:()},config)
    w = h._HGAWorkspace(parents,config,None,patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.zeros(len(r),np.int64) for r in routes]
    args = (routes,bits,w.travel,w.processing,w.points,w.eligible,w.parent_index,
            w.nearest,w.parameters,20,0,0,w.oriented_scratch)
    plain = h._vnd_oriented_arrays(*args,np.zeros(4,np.int64),route_objective=True)
    trace = h.make_vnd_trace(2*len(w.parents),per_move=2)
    observed = h._vnd_oriented_arrays(*args,np.zeros(4,np.int64),route_objective=True,teacher_trace=trace)
    assert plain[2:] == observed[2:]
    for actual,expected in zip(observed[0]+observed[1],plain[0]+plain[1]):
        np.testing.assert_array_equal(actual,expected)
    states,directions,lengths,actions,counts,passes = trace
    accepted = 0
    for slot,pass_number in enumerate(passes):
        if not pass_number:continue
        rr = [states[slot,r,:lengths[slot,r]].copy() for r in range(4)]
        db = [directions[slot,r,:lengths[slot,r]].copy() for r in range(4)]
        assert sorted(int(n) for route in rr for n in route)==sorted(int(n) for route in routes for n in route)
        for rows in actions[slot]:
            for a in rows:
                if a[0]<0:continue
                move,u,v,ru,iu,rv,iv,ordinal,taken = (int(x) for x in a)
                assert rr[ru][iu]==u and rr[rv][iv]==v and ordinal>0
                sa,sb = h._move_segments(rr,move,ru,iu,rv,iv)
                nr,nd = list(rr),list(db)
                nr[ru],nd[ru] = h._materialize_segments(sa,rr),h._materialize_directions(sa,db)
                if move!=4:nr[rv],nd[rv] = h._materialize_segments(sb,rr),h._materialize_directions(sb,db)
                if taken:
                    before=max(h._oriented_route_cost(r,d,w.travel,w.processing) for r,d in zip(rr,db))
                    after=max(h._oriented_route_cost(r,d,w.travel,w.processing) for r,d in zip(nr,nd))
                    assert after<before-1.e-8
                    accepted += 1
    assert accepted>0
    assert max(passes)==plain[4]+1


def test_vnd_teacher_observer_is_optional_and_never_calls_b32(monkeypatch):
    import numpy as np
    from dataclasses import replace
    from mrta_baselines import hga as h,common
    def forbidden(*args,**kwargs):raise AssertionError('unexpected B32/certifier')
    monkeypatch.setattr(common.FormalReferenceEvaluator,'__call__',forbidden)
    monkeypatch.setattr(common,'certify_schedule',forbidden)
    class Observer:
        def __init__(self):self.calls=[]
        def begin(self,w,selected,elapsed,index):return h.make_vnd_trace(2*len(w.parents))
        def end(self,w,selected,trace):self.calls.append(trace[-1].copy())
    observer=Observer()
    settings=replace(h.AdaptedHGAConfig(),vnd_objective='ROUTE_CYCLE',max_iterations=2)
    result=h.run_adapted_hga_v2(_parents(),ScientificConfig(),settings,seed=66,
        time_limit=float('inf'),checkpoints=(),vnd_observer=observer)
    assert result.iterations==2 and len(observer.calls)>0
    assert result.certifier_calls==0


@pytest.mark.parametrize('move',[1,2,3,4,5,6])
def test_vnd_action_context_matches_materialized_oriented_costs(move):
    import numpy as np
    from mrta_baselines import hga as h
    config,parents,solution=_fixture()
    w=h._HGAWorkspace(parents,config,None,patterns=solution.patterns)
    routes=w.encode(solution)
    bits=[np.arange(len(r),dtype=np.int64)%2 for r in routes]
    ru,iu,rv,iv=(0,0,0,2) if move==4 else (0,1,1,1)
    action=np.array([[move,routes[ru][iu],routes[rv][iv],ru,iu,rv,iv,1,0]],np.int64)
    context,affected=h._vnd_candidate_context(routes,bits,action,w.travel,w.processing,
        w.eligible,w.parent_index,w.oriented_scratch)
    a,b=h._move_segments(routes,move,ru,iu,rv,iv)
    rr,dd=list(routes),list(bits)
    rr[ru],dd[ru]=h._materialize_segments(a,routes),h._materialize_directions(a,bits)
    if move!=4:rr[rv],dd[rv]=h._materialize_segments(b,routes),h._materialize_directions(b,bits)
    scale=sum(w.processing[n] for r in routes for n in r)/4
    after=max(h._oriented_route_cost(r,d,w.travel,w.processing) for r,d in zip(rr,dd))
    before=max(h._oriented_route_cost(r,d,w.travel,w.processing) for r,d in zip(routes,bits))
    assert context[0,29]==pytest.approx(after/scale)
    assert context[0,30]==pytest.approx((after-before)/scale)
    assert affected[0,routes[ru][iu]] and affected[0,routes[rv][iv]]


def _br_accounting():
    counts = dict.fromkeys(('reinsertion_attempts','reinsertion_successes','reinsertion_fallbacks',
        'reinsertion_selected_parents','reinsertion_evaluations','reinsertion_transferred_blocks',
        'reinsertion_multi_source_transfers','reinsertion_chain_transfers','reinsertion_pmax_reduced'),0)
    return dict(counts,reinsertion_s=0.,reinsertion_failure_reasons={},reinsertion_examples=[])


def test_br_selection_has_heavy_and_other_movable_sources():
    s,w,a,_ = _saeax_fixture()
    for seed in range(12):
        parents,critical = s.select_reinsertion_parents(a.routes,a.processing,a.costs,w,random.Random(seed))
        assert 3 <= len(parents) <= 4 and len(set(parents))==len(parents)
        sources = {r for r,route in enumerate(a.routes) for n in route
                   if w.parents[w.parent_index[n]].parent_id in parents}
        assert critical in sources and len(sources)>=2
        assert all(any(w.parents[w.parent_index[n]].parent_id==p and
            any(w.eligible[n,t] for t in range(4) if t!=r)
            for r,route in enumerate(a.routes) for n in route) for p in parents)


def test_br_fixed_mode_adapter_reuses_joint_repair_without_scheduling(monkeypatch):
    import numpy as np
    from mrta_search import lns
    from mrta_search import saeax as s
    def forbidden(*args,**kwargs):raise AssertionError('repair must not schedule or use timeline selectors')
    monkeypatch.setattr(s,'_priority_arrays',forbidden)
    monkeypatch.setattr(s,'reference_schedule_formal',forbidden)
    monkeypatch.setattr(lns,'build_crrr_candidate',forbidden)
    monkeypatch.setattr(lns,'select_bottleneck_parents',forbidden)
    parents = (ParentWeld('y',(3.,8.),(3.,4.)),ParentWeld('x',(1.,9.),(10.,9.)),
        ParentWeld('a',(2.,9.),(3.2,9.)),ParentWeld('b',(11.,9.),(12.2,9.)),
        ParentWeld('c',(1.,3.),(2.2,3.)),ParentWeld('d',(12.,3.),(13.2,3.)))
    w = s._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    selected = w.initial_selected.copy()
    xp = next(i for i,p in enumerate(w.parents) if p.parent_id=='x')
    selected[xp] = next(p for p in w.options[xp] if w.patterns[p].kind is SplitKind.X_SPLIT)
    active = {w.ids[n]:int(n) for p in selected for n in w.pattern_nodes[p] if n>=0}
    sequences = (('x::0','y::0','a::whole'),('x::1','b::whole'),('y::1','c::whole'),('d::whole',))
    routes = [np.array([active[n] for n in row],np.int64) for row in sequences]
    bits = [np.zeros(len(r),np.int64) for r in routes]
    s.validate_individual(w,routes,bits,selected)
    monkeypatch.setattr(s,'select_reinsertion_parents',lambda *args:(('x','y','a','c'),0))
    original = s.repair_partial_state_oriented
    calls = []
    def capture(current,directions,partial,config,**kwargs):
        calls.append((current,partial,kwargs))
        assert partial.removed_parent_ids==('a','c','x','y')
        assert partial.routes==((),('b::whole',),(),('d::whole',))
        assert {k:v for k,v in kwargs.items() if k!='seed'}==s.BR_REPAIR_OPTIONS
        return original(current,directions,partial,config,**kwargs)
    monkeypatch.setattr(s,'repair_partial_state_oriented',capture)
    counts = _br_accounting()
    repaired = s.bottleneck_reinsert(routes,bits,selected,w,random.Random(61),counts)
    assert len(calls)==1 and counts['reinsertion_evaluations']<=256
    assert repaired is not None and counts['reinsertion_successes']==1
    rr,dd = repaired
    s.validate_individual(w,rr,dd,selected)
    solution = w.solution(rr,selected)
    assert solution.patterns==calls[0][0].patterns
    assert sorted(n for row in rr for n in row)==sorted(active.values())
    for r,row in enumerate(rr):
        assert all(w.eligible[n,r] for n in row)
    np.testing.assert_array_equal(routes[0],[active[n] for n in sequences[0]])


def test_br_failed_repair_falls_back_to_original_mutation(monkeypatch):
    s,w,a,b = _saeax_fixture()
    settings = s.SAEAXConfig()
    expected = s.adjacency_crossover(a,b,w,random.Random(19),settings)
    monkeypatch.setattr(s,'bottleneck_reinsert',lambda *args:None)
    actual = s.adjacency_crossover(a,b,w,random.Random(19),settings,
        bottleneck_reinsertion=True,accounting={})
    assert s.complete_identity(*actual[:3])==s.complete_identity(*expected[:3])
    assert actual[3:]==expected[3:]


def test_br_one_effective_slot_per_generation_and_one_native_score(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    s,w,_,_ = _saeax_fixture()
    calls,progress = [],[]
    def reinsertion(routes,bits,selected,workspace,rng,accounting):
        accounting['reinsertion_attempts']+=1
        accounting['reinsertion_successes']+=1
        calls.append(accounting['construction_attempts'])
        return routes,bits
    monkeypatch.setattr(s,'bottleneck_reinsert',reinsertion)
    monkeypatch.setattr(s,'_priority_arrays',lambda *args:(100.,None,None,None))
    def b32(solution,config,*,scope,orientations):
        assert progress[-1]['generation']==3
        return ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100.,directions=tuple(orientations[r] for r in range(4)))
    monkeypatch.setattr(s,'reference_schedule_formal',b32)
    parents=tuple(ParentWeld(f'p{i:02d}',(1.+i*1.1,9.),(1.7+i*1.1,9.)) for i in range(16))
    result=s.run_saeax(parents,settings=s.SAEAXConfig(max_generations=3,min_generations=1,
        stagnation_generations=10),seed=987,bottleneck_reinsertion=True,progress_callback=progress.append)
    a=result.accounting
    assert result.generations==3 and result.valid_children==12
    assert a['reinsertion_unique_children']==3
    assert a['native_calls']==a['initialization_evaluations']+12
    assert calls and min(calls)>0 and a['initialization_evaluations']==12
    assert a['b32_calls']==3


def test_br_paired_comparison_excludes_deadlock_and_pilot_requires_load_reduction(monkeypatch):
    import runpy
    from pathlib import Path
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_saeax.py'))
    rows=[dict(layout_id='train_15',seed='20261110',status='FEASIBLE',cmax_s='90',runtime_s='20')]
    ref={('train_15',20261110):dict(status='DEADLOCK',runtime_s='10',cmax_s='')}
    assert m['paired_summary'](rows,ref)['pairs']==0
    old=[dict(layout_id=n,seed='20261110',status='FEASIBLE',cmax_s='100',
        max_processing_s='80',runtime_s='20') for n in ('train_15','train_34','train_28')]
    new=[dict(r,cmax_s='90',runtime_s='25') for r in old]
    monkeypatch.setitem(m['pilot_decision'].__globals__,'method_rows',lambda stage,method:new if method=='br' else old)
    assert not m['pilot_decision']()['passed']
    for r in new[:2]:r['max_processing_s']='79'
    assert m['pilot_decision']()['passed']
    new[2]['runtime_s']='61'
    assert not m['pilot_decision']()['passed']


def test_br_selection_includes_receiver_with_a_legal_third_robot():
    import numpy as np
    from mrta_search import saeax as s
    lengths=(8.,7.,6.,5.,1.,.6,.5)
    parents=tuple(ParentWeld(chr(97+i),(1.,6.),(1.+length,6.)) for i,length in enumerate(lengths))
    w=s._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    nodes=[int(w.pattern_nodes[p,0]) for p in w.initial_selected]
    routes=[np.array(row,np.int64) for row in (nodes[:3],nodes[3:5],nodes[5:6],nodes[6:])]
    bits=[np.zeros(len(r),np.int64) for r in routes]
    costs,loads=s._route_statistics(routes,bits,w.travel,w.processing)
    chosen,critical=s.select_reinsertion_parents(routes,loads,costs,w,random.Random(7))
    assert critical==0 and 'd' in chosen
    heavy=[n for n in routes[critical] if w.parents[w.parent_index[n]].parent_id in chosen]
    receiver=nodes[3]
    assert any(w.eligible[n,1] for n in heavy) and w.eligible[receiver,2]
    assert 3<=len(chosen)<=4


def test_br_completed_pilot_is_read_without_new_runs_or_writes(monkeypatch):
    import runpy
    from pathlib import Path
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_saeax.py'))
    g=m['run_experiment'].__globals__
    layouts=[(n,(None,)*size) for n,size in (('train_15',56),('train_34',90),('train_28',132))]
    saved=[dict(stage='pilot',layout_id=n,seed='20261110',source=m['METHOD_SOURCES']['br']) for n,_ in layouts]
    def forbidden(*args,**kwargs):raise AssertionError('completed experiment was rerun or appended')
    monkeypatch.setitem(g,'load_layouts',lambda split:layouts)
    monkeypatch.setitem(g,'read_rows',lambda:saved)
    monkeypatch.setitem(g,'experiment_job',forbidden)
    monkeypatch.setitem(g,'append_result',forbidden)
    monkeypatch.setitem(g,'summarize',lambda *args:None)
    m['run_experiment']('pilot',1,'br')


def test_br_failed_train_does_not_load_validation_or_run_search(monkeypatch):
    import runpy
    from pathlib import Path
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_saeax.py'))
    g=m['run_experiment'].__globals__
    def forbidden(*args,**kwargs):raise AssertionError('new split touched after failed TRAIN')
    monkeypatch.setitem(g,'pilot_decision',lambda:dict(passed=False))
    monkeypatch.setitem(g,'load_layouts',forbidden)
    monkeypatch.setitem(g,'experiment_job',forbidden)
    with pytest.raises(ValueError,match='TRAIN criteria'):
        m['run_experiment']('validation',1,'br')


def test_eax_nsga_defaults_are_the_requested_natural_budget():
    from mrta_search.eax_nsga2 import EAXNSGAConfig
    s=EAXNSGAConfig()
    assert (s.population_size,s.offspring_per_generation,s.max_generations,
        s.min_generations,s.stagnation_generations,s.retries_per_child)==(20,10,500,100,120,12)
    with pytest.raises(ValueError):EAXNSGAConfig(retries_per_child=0)
    with pytest.raises(ValueError):EAXNSGAConfig(max_generations=10)


def test_eax_table_uses_both_directed_parents_and_ignores_their_owners():
    import numpy as np
    from dataclasses import replace
    from mrta_search import eax_nsga2 as e
    _,w,a,b=_saeax_fixture()
    settings=e.EAXNSGAConfig(adjacency_mutation_probability=0.)
    active,table,parent,detail=e.enhanced_adjacency(a,b,a.selected,w,random.Random(4),settings)
    swapped=replace(b,routes=[b.routes[1],b.routes[0],*b.routes[2:]],
        directions=[b.directions[1],b.directions[0],*b.directions[2:]])
    other=e.enhanced_adjacency(a,swapped,a.selected,w,random.Random(4),settings)
    np.testing.assert_array_equal(table,other[1])
    local={int(n):i for i,n in enumerate(active)}
    u,v=map(int,a.routes[0][:2])
    assert parent[2*local[u],2*local[v]] and parent[2*local[v]+1,2*local[u]+1]
    ds=[w.travel[int(r[k-1]),int(r[k]),int(d[k-1]),int(d[k])]
        for p in (a,b) for r,d in zip(p.routes,p.directions) for k in range(1,len(r))]
    assert detail['distance_threshold_m']==pytest.approx(np.quantile(ds,.75)*w.config.empty_speed)
    selected=a.selected.copy()
    p=next(i for i,options in enumerate(w.options) if len(options)>1)
    selected[p]=next(v for v in w.options[p] if v!=selected[p])
    active2=e.enhanced_adjacency(a,b,selected,w,random.Random(4),settings)[0]
    assert not set(active2).intersection(set(w.pattern_nodes[a.selected[p]])-{-1})


def test_eax_mutation_actually_inserts_directed_neighbours():
    from mrta_search import eax_nsga2 as e
    _,w,a,b=_saeax_fixture()
    active,table,parent,detail=e.enhanced_adjacency(a,b,a.selected,w,random.Random(9),
        e.EAXNSGAConfig(adjacency_mutation_probability=1.))
    assert detail['mutation_entries']>0
    assert table.sum()==detail['threshold_adjacency_entries']+detail['mutation_entries']
    assert parent.sum()-detail['threshold_adjacency_entries']==detail['threshold_excluded_entries']
    assert table.shape==(2*len(active),2*len(active))


def _eax_mixed_fixture():
    import numpy as np
    from mrta_search import eax_nsga2 as e
    parents=(ParentWeld('y',(3.,8.),(3.,4.)),ParentWeld('x',(1.,9.),(10.,9.)),
        ParentWeld('a',(2.,9.),(3.2,9.)),ParentWeld('b',(11.,9.),(12.2,9.)),
        ParentWeld('c',(1.,3.),(2.2,3.)),ParentWeld('d',(12.,3.),(13.2,3.)))
    w=e._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    selected=w.initial_selected.copy()
    xp=next(i for i,p in enumerate(w.parents) if p.parent_id=='x')
    selected[xp]=next(p for p in w.options[xp] if w.patterns[p].kind is SplitKind.X_SPLIT)
    lookup={w.ids[n]:int(n) for p in selected for n in w.pattern_nodes[p] if n>=0}
    def item(sequences):
        rr=[np.array([lookup[n] for n in row],np.int64) for row in sequences]
        dd=[np.zeros(len(r),np.int64) for r in rr]
        costs,loads=e._route_statistics(rr,dd,w.travel,w.processing)
        e.validate_individual(w,rr,dd,selected)
        return e.Individual(rr,dd,selected.copy(),1000.,costs,loads,e.complete_identity(rr,dd,selected),np.empty(0,np.int64))
    a=item((('x::0','y::0','a::whole'),('x::1','b::whole'),('y::1','c::whole'),('d::whole',)))
    b=item((('x::0','a::whole'),('x::1','y::0','b::whole'),('c::whole',),('y::1','d::whole')))
    return e,w,a,b


def test_eax_joint_construction_reallocates_and_preserves_all_split_rules(monkeypatch):
    from mrta_search import saeax,lns
    e,w,a,b=_eax_mixed_fixture()
    def forbidden(*args,**kwargs):raise AssertionError('new EAX delegated construction or scheduled a partial')
    for name in ('adjacency_crossover','assemble_fragments','mutate_child'):
        monkeypatch.setattr(saeax,name,forbidden)
    monkeypatch.setattr(lns,'repair_partial_state_oriented',forbidden)
    monkeypatch.setattr(e,'_priority_arrays',forbidden)
    monkeypatch.setattr(e,'reference_schedule_formal',forbidden)
    monkeypatch.setattr(w,'repair',forbidden)
    settings=e.EAXNSGAConfig(pattern_mutation_probability=0.)
    successes,changed,identities=0,False,set()
    y_owners=set()
    for seed in range(24):
        try:rr,dd,ss,detail=e.adjacency_crossover(a,b,w,random.Random(seed),settings)
        except ValueError:continue
        successes+=1
        e.validate_individual(w,rr,dd,ss)
        assert tuple(ss)==tuple(a.selected)
        actual=w.solution(rr,ss)
        assert actual.patterns==tuple(w.patterns[p] for p in ss)
        owners={w.ids[n]:r for r,row in enumerate(rr) for n in row}
        assert owners['x::0']==0 and owners['x::1']==1
        assert owners['y::0'] in (0,1) and owners['y::1'] in (2,3)
        y_owners.add((owners['y::0'],owners['y::1']))
        changed |= detail['owner_changes_vs_first']>0
        identities.add(e.complete_identity(rr,dd,ss))
        assert detail['few_neighbour_choices']+detail['nearest_fallbacks']==sum(map(len,rr))-1
    assert successes>0 and changed and len(identities)>1 and len(y_owners)>1


def test_eax_nsga_nondominated_ranks_and_constraint_precedence():
    import numpy as np
    from mrta_search.eax_nsga2 import rank_and_crowding
    values=np.array([[10.,5.,9.],[12.,4.,7.],[8.,8.,8.],[20.,10.,20.],
                     [np.inf,0.,0.],[np.inf,-100.,-100.]])
    ranks,crowding=rank_and_crowding(values)
    np.testing.assert_array_equal(ranks,[0,0,0,1,2,2])
    np.testing.assert_array_equal(crowding[-2:],[0.,0.])
    assert np.isfinite(crowding[-2:]).all()


def test_eax_nsga_crowding_is_normalized_and_constant_axes_contribute_zero():
    import numpy as np
    from mrta_search.eax_nsga2 import rank_and_crowding
    values=np.array([[0.,10.,10.],[1.,9.,9.],[2.,8.,8.],[3.,7.,7.]])
    ranks,crowding=rank_and_crowding(values)
    assert (ranks==0).all() and np.isinf(crowding[[0,-1]]).all()
    np.testing.assert_allclose(crowding[1:3],[2.,2.])
    _,crowding=rank_and_crowding(np.array([[0.,1.,3.],[0.,2.,2.],[0.,3.,1.]]))
    assert crowding[1]==pytest.approx(2.)


def test_eax_nsga_environment_retains_nondominated_tradeoffs_and_deduplicates():
    import numpy as np
    from dataclasses import replace
    from mrta_search import eax_nsga2 as e
    _,_,a,_=_saeax_fixture()
    points=[(10.,5.,9.),(12.,4.,7.),(8.,8.,8.),(20.,10.,20.),(float('inf'),0.,0.)]
    pop=[replace(a,native_cmax=c,processing=np.array([p,0,0,0]),
        costs=np.array([p+m,0,0,0]),identity=(i,)) for i,(c,p,m) in enumerate(points)]
    result=e.select_population(pop+[pop[0]],3)
    assert {x.identity for x in result}=={(0,),(1,),(2,)}
    assert e.final_key(pop[2])<e.final_key(pop[0])


@pytest.mark.parametrize('maximum,minimum,stagnation,expected,reason',[
    (8,5,3,5,'STAGNATION'),(3,1,10,3,'MAX_GENERATIONS')])
def test_eax_nsga_natural_stop_and_exactly_one_final_b32(monkeypatch,maximum,minimum,stagnation,expected,reason):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    from mrta_search import eax_nsga2 as e
    calls,native_calls,progress=[],[],[]
    def native(*args):
        native_calls.append(1)
        return 100.,None,None,None
    monkeypatch.setattr(e,'_priority_arrays',native)
    def b32(solution,config,*,scope,orientations):
        assert progress[-1]['generation']==expected
        calls.append(solution)
        return ScheduleResult(ScheduleStatus.FEASIBLE,cmax=777.,directions=tuple(orientations[r] for r in range(4)))
    monkeypatch.setattr(e,'reference_schedule_formal',b32)
    parents=tuple(ParentWeld(f'p{i:02d}',(1.+i*1.1,9.),(1.7+i*1.1,9.)) for i in range(16))
    settings=e.EAXNSGAConfig(population_size=6,offspring_per_generation=3,
        max_generations=maximum,min_generations=minimum,stagnation_generations=stagnation)
    result=e.run_eax_nsga2(parents,settings=settings,seed=987,progress_callback=progress.append)
    assert result.generations==expected and result.valid_children==3*expected
    assert result.termination_reason==reason and result.accounting['search_complete']
    assert result.accounting['native_calls']==len(native_calls)==6+3*expected
    assert result.accounting['b32_calls']==len(calls)==len(result.final_candidates)==1
    assert result.cmax==777. and result.native_cmax==100.
    assert result.runtime>=result.accounting['search_s']+result.accounting['b32_s']


def test_eax_final_b32_deadlock_is_kept_without_fallback(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    from mrta_search import eax_nsga2 as e
    monkeypatch.setattr(e,'_priority_arrays',lambda *args:(100.,None,None,None))
    calls=[]
    def b32(*args,**kwargs):
        calls.append(1)
        return ScheduleResult(ScheduleStatus.DEADLOCK)
    monkeypatch.setattr(e,'reference_schedule_formal',b32)
    parents=tuple(ParentWeld(f'p{i:02d}',(1.+i*1.1,9.),(1.7+i*1.1,9.)) for i in range(16))
    result=e.run_eax_nsga2(parents,settings=e.EAXNSGAConfig(population_size=4,offspring_per_generation=2,
        max_generations=1,min_generations=1),seed=987)
    assert result.status=='DEADLOCK' and result.cmax is None and result.native_cmax==100.
    assert result.accounting['b32_calls']==len(calls)==1
    assert result.final_candidates[0]['cmax_s'] is None and result.accounting['wait_is_partial']


def test_eax_duplicate_child_retries_are_per_slot_and_do_not_complete_a_generation(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    from mrta_search import eax_nsga2 as e
    _,w,a,b=_saeax_fixture()
    detail=dict.fromkeys(('distance_threshold_m','inherited_edges','new_edges','mutation_entries',
        'threshold_adjacency_entries','threshold_excluded_entries','parental_adjacency_entries','few_neighbour_choices','nearest_fallbacks',
        'deferred_nodes','owner_changes_vs_first','owner_changes_vs_second'),0)
    calls=[]
    def crossed(*args):
        calls.append(1)
        item=b if len(calls)==2 else a
        return item.routes,item.directions,item.selected.copy(),dict(detail)
    monkeypatch.setattr(e,'adjacency_crossover',crossed)
    monkeypatch.setattr(e,'_priority_arrays',lambda *args:(100.,None,None,None))
    monkeypatch.setattr(e,'reference_schedule_formal',lambda *args,orientations,**kwargs:
        ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100.,directions=tuple(orientations[r] for r in range(4))))
    result=e.run_eax_nsga2(w.parents,settings=e.EAXNSGAConfig(population_size=2,offspring_per_generation=2,
        max_generations=2,min_generations=1),seed=7)
    assert len(calls)==2+12 and result.accounting['construction_attempts']==12
    assert result.generations==0 and result.valid_children==0
    assert result.termination_reason=='NOVEL_CHILD_RETRIES_EXHAUSTED'
    assert not result.accounting['search_complete'] and result.accounting['native_calls']==2


def test_eax_pilot_uses_b32_quality_and_time_with_pmax_diagnostic_only(monkeypatch):
    import runpy,json
    from pathlib import Path
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_eax_nsga2.py'))
    old={ (n,20261110):dict(status='FEASIBLE',cmax_s='100',runtime_s='20') for n in m['PILOTS']}
    rows=[dict(layout_id=n,seed='20261110',status='FEASIBLE',cmax_s='90',runtime_s='50',
        max_processing_s='10000',diagnostic=json.dumps(dict(accounting=dict(search_complete=True)))) for n in m['PILOTS']]
    g=m['pilot_decision'].__globals__
    monkeypatch.setitem(g,'method_rows',lambda stage:rows)
    monkeypatch.setitem(g,'saeax_rows',lambda stage:old)
    assert m['pilot_decision']()['passed']
    rows[0]['status']='DEADLOCK';rows[0]['cmax_s']=''
    assert not m['pilot_decision']()['passed']
    rows[0]['status']='FEASIBLE';rows[0]['cmax_s']='90'
    for r in rows:r['runtime_s']='104'
    assert not m['pilot_decision']()['passed']


def test_eax_failed_train_prevents_further_split_access(monkeypatch):
    import runpy
    from pathlib import Path
    m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_eax_nsga2.py'))
    g=m['run_experiment'].__globals__
    def forbidden(*args,**kwargs):raise AssertionError('new split/search accessed after failed TRAIN')
    monkeypatch.setitem(g,'pilot_decision',lambda:dict(passed=False))
    monkeypatch.setitem(g,'load_layouts',forbidden)
    monkeypatch.setitem(g,'experiment_job',forbidden)
    with pytest.raises(ValueError,match='TRAIN screening'):
        m['run_experiment']('validation',1)


def test_eax_threshold_matches_gao_short_and_both_long_cases():
    import numpy as np
    from types import SimpleNamespace
    from mrta_search import eax_nsga2 as e
    travel=np.zeros((10,10,2,2))
    for i,d in enumerate((1.,1.,1.,1.,1.,100.,101.,1.,1.)):travel[i,i+1,0,0]=d
    w=SimpleNamespace(pattern_nodes=np.column_stack((np.arange(10),np.full(10,-1))),
        ids=tuple(map(str,range(10))),parent_index=np.arange(10),travel=travel,
        config=SimpleNamespace(empty_speed=.2))
    p=SimpleNamespace(routes=[np.arange(10),[],[],[]],directions=[np.zeros(10,np.int64),[],[],[]])
    _,table,_,detail=e.enhanced_adjacency(p,p,np.arange(10),w,random.Random(4),
        e.EAXNSGAConfig(adjacency_mutation_probability=0.))
    assert detail['distance_threshold_m']==pytest.approx(.2)
    assert not table[2*5,2*6]  # 1/100: discard the long successor
    assert table[2*6+1,2*5+1] and table[2*6,2*7]  # 100/101: retain both
    assert not table[2*7+1,2*6+1]  # 101/1: discard the long predecessor
    assert table[2*2+1,2*1+1] and table[2*2,2*3]  # both short retained
    assert table[2*9+1,2*8+1]  # open-route endpoint retained


def test_eax_initialization_exhaustion_preserves_actual_evaluation_counts(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    from mrta_search import eax_nsga2 as e
    _,w,a,_=_saeax_fixture()
    detail=dict.fromkeys(('distance_threshold_m','inherited_edges','new_edges','mutation_entries',
        'threshold_adjacency_entries','threshold_excluded_entries','parental_adjacency_entries',
        'few_neighbour_choices','nearest_fallbacks','deferred_nodes','owner_changes_vs_first','owner_changes_vs_second'),0)
    monkeypatch.setattr(e,'adjacency_crossover',lambda *args:(a.routes,a.directions,a.selected.copy(),dict(detail)))
    monkeypatch.setattr(e,'_priority_arrays',lambda *args:(100.,None,None,None))
    monkeypatch.setattr(e,'reference_schedule_formal',lambda *args,orientations,**kwargs:
        ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100.,directions=tuple(orientations[r] for r in range(4))))
    result=e.run_eax_nsga2(w.parents,settings=e.EAXNSGAConfig(population_size=2,
        max_generations=1,min_generations=1),seed=7)
    assert result.termination_reason=='INITIALIZATION_RETRIES_EXHAUSTED'
    assert not result.accounting['search_complete']
    assert result.accounting['initialization_attempts']==13 and result.accounting['native_calls']==1
    assert result.accounting['initialization_evaluations']==1 and result.accounting['duplicate_children']==12
    assert result.generations==result.valid_children==0 and result.accounting['b32_calls']==1


def test_eax_joint_position_processing_and_move_match_full_route_recomputation():
    import numpy as np
    from mrta_search import eax_nsga2 as e
    from mrta_search.saeax import _heads_legal
    _,w,a,_=_saeax_fixture()
    active=np.array(sorted(int(n) for p in a.selected for n in w.pattern_nodes[p] if n>=0),np.int64)
    node=int(active[3])
    rows=[np.array([n for n in active[:6] if n!=node],np.int64),active[6:].copy(),
          np.empty(0,np.int64),np.empty(0,np.int64)]
    directions=[np.arange(len(r),dtype=np.int64)%2 for r in rows]
    loads_and_costs=e._route_statistics(rows,directions,w.travel,w.processing)
    costs,loads=loads_and_costs
    rr,dd=np.zeros((4,len(active)),np.int64),np.zeros((4,len(active)),np.int64)
    lengths=np.array([len(r) for r in rows],np.int64)
    for r in range(4):rr[r,:lengths[r]],dd[r,:lengths[r]]=rows[r],directions[r]
    local=np.full(len(w.ids),-1,np.int64);local[active]=np.arange(len(active))
    table=np.zeros((2*len(active),2*len(active)),np.bool_)
    r,k,d,delta=e._joint_position(node,0,local,table,rr,dd,lengths,loads,costs,
        w.travel,w.processing,w.eligible,w.parent_index,w.points,w.parameters)
    assert r==1  # R0 has five blocks, R1 has two: release the heavy route.
    possible=[]
    chosen=None
    for robot in range(4):
        if not w.eligible[node,robot]:continue
        for position in range(len(rows[robot])+1):
            for bit in range(2):
                nr,nd=list(rows),list(directions)
                nr[robot]=np.insert(nr[robot],position,node)
                nd[robot]=np.insert(nd[robot],position,bit)
                if not _heads_legal(nr,nd,w.points,w.parameters):continue
                nc,np_=e._route_statistics(nr,nd,w.travel,w.processing)
                dm=float(sum(nc-np_)-sum(costs-loads))
                score=(float(max(np_)),float(max(nc)),dm)
                possible.append(score)
                if (robot,position,bit)==(r,k,d):chosen=score
    assert chosen is not None
    assert chosen==pytest.approx(min(possible))
    assert delta==pytest.approx(chosen[2])


def _bc_initial_fixture():
    from mrta_search import bc_ecs as c
    from mrta_search.saeax import SAEAXResult
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    _,w,a,b=_saeax_fixture()
    solution=w.solution(a.routes,a.selected)
    vectors=tuple(tuple(int(d) for d in row) for row in a.directions)
    schedule=ScheduleResult(ScheduleStatus.FEASIBLE,cmax=1000.,directions=vectors)
    accounting=dict(native_calls=12,b32_calls=3,b32_s=0.,search_s=0.,
        processing_s=a.processing.tolist(),move_s=float(sum(a.costs-a.processing)),wait_s=0.)
    initial=SAEAXResult('FEASIBLE',1000.,solution,vectors,schedule,900.,0,0,'STAGNATION',0.,accounting)
    return c,w,a,b,initial


def _bc_fake_candidates(c,w,item,count=6):
    import numpy as np
    for index in range(count):
        rr=[r.copy() for r in item.routes]
        dd=[d.copy() for d in item.directions]
        for bit in range(3):
            if (index+1)&(1<<bit):dd[0][bit+1]^=1
        cc,pp=c._route_statistics(rr,dd,w.travel,w.processing)
        key=c.complete_identity(rr,dd,item.selected)
        yield rr,dd,cc,pp,key,dict(kind='TWO_ROBOT_EXCHANGE',parents=['a','e'],
            transfers=[],variant=0,wait_anchored=False)


def test_bc_default_budget_and_conservative_bound_comparisons():
    from mrta_search.bc_ecs import BCECSConfig,cannot_improve
    assert (BCECSConfig().max_candidates,BCECSConfig().stagnation_candidates)==(6000,1500)
    with pytest.raises(ValueError):BCECSConfig(max_candidates=0)
    assert not cannot_improve(100.,100.,100)
    assert not cannot_improve(100.+1e-9,100.,100)
    assert cannot_improve(101.,100.,100)
    assert not cannot_improve(100.,float('inf'),100)


def test_bc_assignment_processing_is_exact_for_joint_exchange():
    import numpy as np
    c,w,a,_,_=_bc_initial_fixture()
    owners=np.full(len(w.ids),-1,np.int64)
    for r,row in enumerate(a.routes):owners[row]=r
    nodes=np.array([a.routes[0][1],a.routes[1][2]],np.int64)
    targets=np.array([1,0],np.int64)
    projected=c.assignment_loads(a.processing,nodes,owners,targets,w.processing)
    rows=[r[~np.isin(r,nodes)].copy() for r in a.routes]
    for node,target in zip(nodes,targets):rows[target]=np.append(rows[target],node)
    _,actual=c._route_statistics(rows,[np.zeros(len(r),np.int64) for r in rows],w.travel,w.processing)
    np.testing.assert_allclose(projected,actual,atol=1e-9)


def test_bc_chain_plans_include_two_robot_exchange_and_three_robot_cycle():
    import itertools,numpy as np
    from mrta_search import bc_ecs as c
    parents=tuple(ParentWeld(chr(97+i),(1.+2*i,6.),(2.+2*i,6.)) for i in range(9))
    w=c._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    nodes=[int(w.pattern_nodes[p,0]) for p in w.initial_selected]
    rows=[np.array(r,np.int64) for r in (nodes[:4],nodes[4:6],nodes[6:8],nodes[8:])]
    bits=[np.zeros(len(r),np.int64) for r in rows]
    costs,loads=c._route_statistics(rows,bits,w.travel,w.processing)
    item=c.Individual(rows,bits,w.initial_selected.copy(),10000.,costs,loads,
        c.complete_identity(rows,bits,w.initial_selected),np.empty(0,np.int64))
    plans=list(itertools.islice(c.chain_plans(item,w,np.zeros(len(parents)),random.Random(3)),500))
    assert any(len(p)==1 for p in plans)
    assert any(len(p)==2 and p[-1][2]==p[0][1] for p in plans)
    assert any(len(p)==3 and p[-1][2]==p[0][1] and len({a[1] for a in p})==3 for p in plans)
    for plan in plans:
        assert plan[0][1] in (int(np.argmax(loads)),int(np.argmax(costs)))
        assert len({int(w.parent_index[n]) for n,_,_ in plan})==len(plan)
        for node,source,target in plan:
            assert node in rows[source] and target!=source and w.eligible[node,target]
        for a,b in zip(plan,plan[1:]):assert a[2]==b[1]


def test_bc_parent_closed_repair_keeps_y_independent_and_x_fixed(monkeypatch):
    import numpy as np
    from mrta_search import bc_ecs as c
    parents=(ParentWeld('y',(3.,8.),(3.,4.)),ParentWeld('x',(1.,9.),(10.,9.)),
        ParentWeld('a',(2.,9.),(3.2,9.)),ParentWeld('b',(11.,9.),(12.2,9.)),
        ParentWeld('c',(1.,3.),(2.2,3.)),ParentWeld('d',(12.,3.),(13.2,3.)))
    w=c._HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    selected=w.initial_selected.copy()
    xp=next(i for i,p in enumerate(w.parents) if p.parent_id=='x')
    selected[xp]=next(p for p in w.options[xp] if w.patterns[p].kind is SplitKind.X_SPLIT)
    lookup={w.ids[n]:int(n) for p in selected for n in w.pattern_nodes[p] if n>=0}
    sequences=(('x::0','y::0','a::whole'),('x::1','b::whole'),('y::1','c::whole'),('d::whole',))
    rows=[np.array([lookup[n] for n in r],np.int64) for r in sequences]
    bits=[np.zeros(len(r),np.int64) for r in rows]
    costs,loads=c._route_statistics(rows,bits,w.travel,w.processing)
    item=c.Individual(rows,bits,selected,10000.,costs,loads,c.complete_identity(rows,bits,selected),np.empty(0,np.int64))
    solution=w.solution(rows,selected)
    chain=((lookup['y::0'],0,1),(lookup['b::whole'],1,0))
    assignments=list(c.chain_assignments(chain,item,w))
    assert {int(targets[list(nodes).index(lookup['y::1'])]) for _,nodes,targets in assignments}=={2,3}
    captured=[]
    original=c._remove_parent_set
    def remove(current,parent_ids):
        partial=original(current,parent_ids);captured.append(partial)
        assert set(parent_ids)=={'b','y'}
        assert not any(n.startswith(('b::','y::')) for r in partial.routes for n in r)
        return partial
    monkeypatch.setattr(c,'_remove_parent_set',remove)
    successes=0
    for ps,nodes,targets in assignments:
        for variant in (0,1):
            repaired=c.construct_chain(item,solution,w,ps,nodes,targets,chain,variant)
            if repaired is None:continue
            rr,dd=repaired;successes+=1
            c.validate_individual(w,rr,dd,selected)
            owners={w.ids[n]:r for r,row in enumerate(rr) for n in row}
            assert owners['x::0']==0 and owners['x::1']==1
            assert owners['y::0']==1 and owners['b::whole']==0
            assert w.solution(rr,selected).patterns==solution.patterns
    assert successes>0 and captured
    xn=np.array([lookup['x::0'],lookup['x::1']],np.int64)
    with pytest.raises(ValueError,match='fixed X'):
        c.construct_chain(item,solution,w,(xp,),xn,np.array([1,0],np.int64),((xn[0],0,1),),0)


def test_bc_wait_feedback_uses_observed_native_wait_only():
    import numpy as np
    c,w,a,_,_=_bc_initial_fixture()
    node=int(a.routes[0][0]);ops=np.zeros((2,9))
    ops[0,0],ops[0,7],ops[0,8]=12.5,node,4
    ops[1,0],ops[1,7],ops[1,8]=99.,node,0
    actual=c.observed_native_waits([ops],w)
    assert sum(actual)==12.5 and actual[w.parent_index[node]]==12.5


@pytest.mark.parametrize('status,cmax,reason',[
    ('FEASIBLE',850.,'IMPROVED_B32'),('FEASIBLE',1100.,'INITIAL_B32_BETTER_OR_EQUAL'),
    ('DEADLOCK',None,'FINAL_B32_FAILED')])
def test_bc_one_in_run_seed_and_one_final_b32_with_cached_fallback(monkeypatch,status,cmax,reason):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    c,w,a,_,initial=_bc_initial_fixture()
    seeds,finals,native=[],[],[]
    def seed(parents,config,**kwargs):
        seeds.append(kwargs)
        return initial
    monkeypatch.setattr(c,'run_saeax',seed)
    monkeypatch.setattr(c,'chain_candidates',lambda item,solution,workspace,*args:
        _bc_fake_candidates(c,workspace,item,1))
    def scored(*args):
        native.append(1)
        return 800.,None,[],None
    monkeypatch.setattr(c,'_priority_arrays',scored)
    def b32(solution,config,*,scope,orientations):
        finals.append(solution)
        return ScheduleResult(ScheduleStatus(status),cmax=cmax,directions=tuple(orientations[r] for r in range(4)))
    monkeypatch.setattr(c,'reference_schedule_formal',b32)
    result=c.run_bc_ecs(w.parents,settings=c.BCECSConfig(max_candidates=1,stagnation_candidates=10),seed=99)
    assert len(seeds)==len(native)==len(finals)==1 and seeds[0]['seed']==99
    assert result.accounting['native_calls']==13 and result.accounting['bc_native_calls']==1
    assert result.accounting['b32_calls']==4 and result.accounting['final_b32_calls']==1
    assert result.accounting['return_reason']==reason and result.status=='FEASIBLE'
    assert result.cmax==(850. if reason=='IMPROVED_B32' else 1000.)
    if reason!='IMPROVED_B32':assert result.schedule is initial.schedule
    if status=='DEADLOCK':assert result.final_candidate['cmax_s'] is None
    assert result.runtime>=result.accounting['initialization_s']+result.accounting['chain_search_s']+result.accounting['final_b32_s']


@pytest.mark.parametrize('value',[950.,float('inf')])
def test_bc_stagnation_counts_actual_evaluations_and_native_rejection_is_not_b32_deadlock(monkeypatch,value):
    c,w,_,_,initial=_bc_initial_fixture()
    monkeypatch.setattr(c,'run_saeax',lambda *args,**kwargs:initial)
    monkeypatch.setattr(c,'chain_candidates',lambda item,solution,workspace,*args:
        _bc_fake_candidates(c,workspace,item,6))
    monkeypatch.setattr(c,'_priority_arrays',lambda *args:(value,None,[],None))
    def forbidden(*args,**kwargs):raise AssertionError('unimproved candidate or initial was scheduled again')
    monkeypatch.setattr(c,'reference_schedule_formal',forbidden)
    result=c.run_bc_ecs(w.parents,settings=c.BCECSConfig(max_candidates=10,stagnation_candidates=3))
    assert result.termination_reason=='STAGNATION' and result.accounting['bc_native_calls']==3
    assert result.accounting['stagnation_at_stop']==3 and result.accounting['b32_calls']==3
    assert result.cmax==1000. and result.schedule is initial.schedule
    assert result.accounting['native_nonfinite_candidates']==(3 if value==float('inf') else 0)
    assert result.final_candidate is None


def test_bc_initialization_failure_does_not_run_a_chain_or_extra_b32(monkeypatch):
    c,w,_,_,initial=_bc_initial_fixture()
    initial.status='DEADLOCK';initial.cmax=None
    monkeypatch.setattr(c,'run_saeax',lambda *args,**kwargs:initial)
    def forbidden(*args,**kwargs):raise AssertionError('search started without a feasible initial schedule')
    monkeypatch.setattr(c,'chain_candidates',forbidden)
    monkeypatch.setattr(c,'reference_schedule_formal',forbidden)
    result=c.run_bc_ecs(w.parents)
    assert result.termination_reason=='INITIALIZATION_FAILED' and result.cmax is None
    assert result.accounting['bc_native_calls']==0 and result.accounting['b32_calls']==3


def _bc_candidate_counts():
    return dict.fromkeys(('chain_plans','assignment_combinations','processing_pruned','route_bound_pruned',
        'construction_rejected','complete_constructed'),0)


def test_bc_processing_bound_prunes_before_construction_and_does_not_fake_evaluations(monkeypatch):
    import numpy as np
    c,w,a,_,initial=_bc_initial_fixture()
    def forbidden(*args,**kwargs):raise AssertionError('processing-pruned plan was constructed or scored')
    monkeypatch.setattr(c,'construct_chain',forbidden)
    monkeypatch.setattr(c,'_priority_arrays',forbidden)
    counts=_bc_candidate_counts()
    candidates=list(c.chain_candidates(a,initial.solution,w,np.zeros(len(w.parents)),random.Random(3),1.,counts))
    assert candidates==[] and counts['processing_pruned']>0 and counts['complete_constructed']==0


def test_bc_complete_directed_route_bound_prunes_without_native_or_b32(monkeypatch):
    import numpy as np
    c,w,a,_,initial=_bc_initial_fixture()
    def forbidden(*args,**kwargs):raise AssertionError('complete-bound pruning called an evaluator')
    monkeypatch.setattr(c,'_priority_arrays',forbidden)
    monkeypatch.setattr(c,'reference_schedule_formal',forbidden)
    counts=_bc_candidate_counts()
    target=float(max(a.processing))+1.
    candidates=list(c.chain_candidates(a,initial.solution,w,np.zeros(len(w.parents)),random.Random(3),target,counts))
    assert candidates==[] and counts['complete_constructed']>0
    assert counts['route_bound_pruned']==counts['complete_constructed']


def test_bc_real_complete_candidate_bounds_do_not_exceed_finite_native_cmax():
    import itertools,numpy as np
    c,w,a,_,initial=_bc_initial_fixture()
    counts=_bc_candidate_counts()
    candidates=list(itertools.islice(c.chain_candidates(a,initial.solution,w,
        np.zeros(len(w.parents)),random.Random(3),1500.,counts),12))
    assert candidates
    for rr,dd,cc,pp,_,_ in candidates:
        native,_,_,_=c._priority_arrays(rr,w.travel,w.processing,w.points,w.parameters,dd)
        if np.isfinite(native):
            assert max(pp)<=max(cc)+1e-7 and max(cc)<=native+1e-7


def test_bc_entry_train_criteria_use_actual_b32_and_complete_time(monkeypatch):
    import json
    from mrta_search import bc_ecs as b
    old={(name,20261110):dict(status='FEASIBLE',cmax_s='100',runtime_s='20') for name in b.PILOTS}
    rows=[dict(layout_id=name,seed='20261110',status='FEASIBLE',cmax_s=str(score),runtime_s='50',
        diagnostic=json.dumps(dict(accounting=dict(search_complete=True))))
        for name,score in zip(b.PILOTS,(90.,95.,100.))]
    monkeypatch.setattr(b,'method_rows',lambda stage:rows)
    monkeypatch.setattr(b,'saeax_rows',lambda stage:old)
    assert b.pilot_decision()['passed']
    rows[0]['status']='DEADLOCK';rows[0]['cmax_s']=''
    assert not b.pilot_decision()['passed']
    rows[0]['status']='FEASIBLE';rows[0]['cmax_s']='90'
    for r in rows:r['runtime_s']='104'
    assert not b.pilot_decision()['passed']


def test_bc_entry_failed_train_does_not_access_validation(monkeypatch):
    from mrta_search import bc_ecs as b
    def forbidden(*args,**kwargs):raise AssertionError('VALIDATION accessed without passed TRAIN')
    monkeypatch.setattr(b,'pilot_decision',lambda:dict(passed=False))
    monkeypatch.setattr(b,'load_layouts',forbidden)
    monkeypatch.setattr(b,'experiment_job',forbidden)
    with pytest.raises(ValueError,match='TRAIN screening'):
        b.run_experiment('validation',1)


def test_bc_entry_skips_saved_runs_without_writes_or_initialization(monkeypatch):
    from mrta_search import bc_ecs as b
    layouts=[(name,()) for name in b.PILOTS]
    saved=[dict(layout_id=name,seed='20261110') for name in b.PILOTS]
    def forbidden(*args,**kwargs):raise AssertionError('saved BC result was repeated or appended')
    monkeypatch.setattr(b,'load_layouts',lambda split:layouts)
    monkeypatch.setattr(b,'method_rows',lambda stage:saved)
    monkeypatch.setattr(b,'experiment_job',forbidden)
    monkeypatch.setattr(b,'append_result',forbidden)
    monkeypatch.setattr(b,'summarize',lambda stage:None)
    b.run_experiment('pilot',1)


@pytest.mark.parametrize('node_offset', [0, 2, 5, 7])
def test_rcig_joint_insertion_matches_existing_action_pool_and_full_cost(node_offset):
    import numpy as np
    from mrta_search import rc_ig as c
    from mrta_search.lns import joint_block_candidates, oriented_route_state, oriented_time_score
    _, w, item, _ = _saeax_fixture()
    node = int(np.concatenate(item.routes)[node_offset])
    rows = [r[r != node].copy() for r in item.routes]
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in rows]
    solution = w.solution(item.routes, item.selected)
    blocks = block_map(solution, w.config)
    state = oriented_route_state(tuple(tuple(w.ids[n] for n in r) for r in rows),
        tuple(tuple(int(d) for d in r) for r in bits), blocks, w.config)
    pattern = w.patterns[int(w.node_pattern[node])]
    reference = joint_block_candidates(state, blocks[w.ids[node]], pattern, blocks, w.config,
        insertion_limit=sum(map(len, rows))+1)
    by_position = {}
    for after, action in reference:
        score = oriented_time_score(after)
        key = action.robot, action.position
        if key not in by_position or score < by_position[key]:
            by_position[key] = score
    actual = c.insertion_options(node, rows, bits, w, dict(insertion_states=0))
    assert len(actual) == min(2, len(by_position))
    for (rr, dd, score, actions), expected in zip(actual, sorted(by_position.values())):
        cc, pp = c._route_statistics(rr, dd, w.travel, w.processing)
        assert score == pytest.approx((max(cc), sum(cc)), abs=1e-8)
        assert score == pytest.approx(expected, abs=1e-7)
        c.validate_individual(w, rr, dd, item.selected)
        assert c._heads_legal(rr, dd, w.points, w.parameters)
        action = actions[0]
        r, k = action['robot'], action['position']
        assert rr[r][k] == node and dd[r][k] == action['direction']
        for name, old, new in action['neighbour_changes']:
            assert old != new
            assert dd[r][list(w.ids[n] for n in rr[r]).index(name)] == new


def test_rcig_catalog_modes_parent_coverage_y_qualification_and_fixed_x():
    import numpy as np
    from mrta_search import rc_ig as c
    parents = (ParentWeld('y', (3.,8.), (3.,4.)), ParentWeld('x', (1.,9.), (10.,9.)),
        ParentWeld('a', (2.,9.), (3.2,9.)), ParentWeld('b', (11.,9.), (12.2,9.)),
        ParentWeld('c', (1.,3.), (2.2,3.)), ParentWeld('d', (12.,3.), (13.2,3.)))
    w = c._HGAWorkspace(parents, ScientificConfig(), FORMAL_SCOPE_V2)
    selected = w.initial_selected.copy()
    xp = next(i for i,p in enumerate(w.parents) if p.parent_id == 'x')
    selected[xp] = next(p for p in w.options[xp] if w.patterns[p].kind is SplitKind.X_SPLIT)
    lookup = {w.ids[n]:int(n) for p in selected for n in w.pattern_nodes[p] if n >= 0}
    names = (('a::whole','x::0','y::0'), ('b::whole','x::1'), ('c::whole','y::1'), ('d::whole',))
    rows = [np.array([lookup[n] for n in r], np.int64) for r in names]
    bits = [np.zeros(len(r), np.int64) for r in rows]
    costs, loads = c._route_statistics(rows, bits, w.travel, w.processing)
    item = c.Individual(rows, bits, selected, 10000., costs, loads,
        c.complete_identity(rows,bits,selected), np.empty(0,np.int64))
    kinds = set()
    original_options = w.options
    for parent in (xp, next(i for i,p in enumerate(w.parents) if p.parent_id == 'y')):
        for pattern in original_options[parent]:
            w.options = tuple((pattern,) if i == parent else v for i,v in enumerate(original_options))
            a = dict(insertion_states=0, mode_trials=0)
            result = c.repair_parents(item, (parent,), w, a)
            assert result is not None
            rr, dd, ss, trace = result
            c.validate_individual(w,rr,dd,ss)
            assert ss[parent] == pattern
            kinds.add(w.patterns[pattern].kind)
            solution = w.solution(rr,ss)
            assert tuple(r.block_ids for r in solution.routes) == tuple(tuple(w.ids[n] for n in r) for r in rr)
            for r,row in enumerate(rr):
                for n in row:
                    assert w.eligible[n,r]
                    if w.patterns[w.node_pattern[n]].kind is SplitKind.X_SPLIT:
                        assert r == (0 if w.patterns[w.node_pattern[n]].rail.value == 'UPPER' else 2)+int(w.node_child[n])-1
            assert len(trace) == 1 and trace[0]['parent_id'] == w.parents[parent].parent_id
    assert kinds == {SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT}
    w.options = original_options
    repaired = c.repair_parents(item, (xp, 0, 1), w, dict(insertion_states=0, mode_trials=0))
    assert repaired is not None
    c.validate_individual(w, *repaired[:3])


@pytest.mark.parametrize('status,cmax,reason', [
    ('FEASIBLE', 850., 'IMPROVED_B32'), ('FEASIBLE', 1100., 'INITIAL_B32_BETTER_OR_EQUAL'),
    ('DEADLOCK', None, 'FINAL_B32_FAILED')])
def test_rcig_500_attempts_distinct_evaluations_one_b32_and_cached_fallback(monkeypatch,status,cmax,reason):
    from types import SimpleNamespace
    from mrta_search import rc_ig as c
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    _, w, item, _, initial = _bc_initial_fixture()
    calls = dict(seed=0, repair=0, native=0, final=0)
    def seed(*args, **kwargs):
        calls['seed'] += 1
        assert kwargs['seed'] == 99
        return initial
    fake = next(_bc_fake_candidates(c,w,item,1))
    def repair(*args):
        calls['repair'] += 1
        return None if calls['repair'] % 2 == 0 else (fake[0],fake[1],item.selected.copy(),[])
    def score(*args):
        calls['native'] += 1
        return 800.,None,[],None
    def final(*args, orientations, **kwargs):
        calls['final'] += 1
        return ScheduleResult(ScheduleStatus(status),cmax=cmax,
            directions=tuple(orientations[r] for r in range(4)))
    elapsed = [0.]
    def clock():
        elapsed[0] += 10.
        return elapsed[0]
    monkeypatch.setattr(c,'time',SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(c,'run_saeax',seed)
    monkeypatch.setattr(c,'repair_parents',repair)
    monkeypatch.setattr(c,'_priority_arrays',score)
    monkeypatch.setattr(c,'reference_schedule_formal',final)
    result = c.run_rc_ig(w.parents,seed=99)
    a = result.accounting
    assert calls == dict(seed=1,repair=500,native=1,final=1)
    assert result.iterations == 500 and result.termination_reason == 'ITERATION_LIMIT'
    assert a['construction_failures'] == 250 and a['duplicate_candidates'] == 249
    assert a['native_calls'] == 13 and a['ig_native_calls'] == 1
    assert a['b32_calls'] == 4 and a['final_b32_calls'] == 1
    assert result.runtime > 90 and result.runtime >= a['initialization_s']+a['ig_search_s']+a['final_b32_s']
    assert a['return_reason'] == reason and a['search_complete']
    if reason == 'IMPROVED_B32':
        assert result.cmax == 850. and result.native_cmax == 800.
    else:
        assert result.schedule is initial.schedule and result.cmax == initial.cmax
    if status == 'DEADLOCK':
        assert result.final_candidate['cmax_s'] is None


def test_rcig_annealing_accepts_worse_rejects_infeasible_and_returns_global_best(monkeypatch):
    from mrta_search import rc_ig as c
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    _, w, item, _, initial = _bc_initial_fixture()
    candidates = iter(_bc_fake_candidates(c,w,item,3))
    values = iter((800.,850.,float('inf')))
    monkeypatch.setattr(c,'run_saeax',lambda *args,**kwargs:initial)
    def repaired(*args):
        f = next(candidates)
        return f[0], f[1], item.selected.copy(), []
    class RNG:
        def __init__(self,*args):pass
        def random(self):return 0.
    monkeypatch.setattr(c.random,'Random',RNG)
    monkeypatch.setattr(c,'destroy_parents',lambda *args:(0,1,2))
    monkeypatch.setattr(c,'repair_parents',repaired)
    monkeypatch.setattr(c,'_priority_arrays',lambda *args:(next(values),None,[],None))
    monkeypatch.setattr(c,'reference_schedule_formal',lambda *args,orientations,**kwargs:
        ScheduleResult(ScheduleStatus.FEASIBLE,cmax=900.,directions=tuple(orientations[r] for r in range(4))))
    result = c.run_rc_ig(w.parents,settings=c.RCIGConfig(max_iterations=3))
    assert result.accounting['accepted_improvements'] == 1
    assert result.accounting['accepted_worse'] == 1
    assert result.accounting['native_nonfinite_candidates'] == 1
    assert result.native_cmax == 800. and result.cmax == 900.
    assert result.progress[-1]['current_native_cmax_s'] == 850.


@pytest.mark.parametrize('value',[950.,float('inf')])
def test_rcig_no_global_native_improvement_never_reschedules_initial(monkeypatch,value):
    from mrta_search import rc_ig as c
    _,w,item,_,initial = _bc_initial_fixture()
    f = next(_bc_fake_candidates(c,w,item,1))
    monkeypatch.setattr(c,'run_saeax',lambda *args,**kwargs:initial)
    monkeypatch.setattr(c,'repair_parents',lambda *args:(f[0],f[1],item.selected.copy(),[]))
    monkeypatch.setattr(c,'_priority_arrays',lambda *args:(value,None,[],None))
    def forbidden(*args,**kwargs):raise AssertionError('unexpected B32')
    monkeypatch.setattr(c,'reference_schedule_formal',forbidden)
    result = c.run_rc_ig(w.parents,settings=c.RCIGConfig(max_iterations=2))
    assert result.schedule is initial.schedule and result.accounting['final_b32_calls'] == 0
    assert result.accounting['return_reason'] == 'NO_NATIVE_IMPROVEMENT'


def test_rcig_defaults_initial_failure_and_destroy_size(monkeypatch):
    import numpy as np
    from mrta_search import rc_ig as c
    _,w,item,_,initial = _bc_initial_fixture()
    assert c.RCIGConfig().max_iterations == 500
    with pytest.raises(ValueError):c.RCIGConfig(max_iterations=501)
    removed = c.destroy_parents(item,w,np.ones(len(w.parents)),np.zeros(4),random.Random(8))
    assert len(removed) == len(set(removed)) == 3
    initial.status = 'DEADLOCK';initial.cmax = None
    monkeypatch.setattr(c,'run_saeax',lambda *args,**kwargs:initial)
    def forbidden(*args,**kwargs):raise AssertionError('search without feasible seed')
    monkeypatch.setattr(c,'repair_parents',forbidden)
    result = c.run_rc_ig(w.parents)
    assert result.iterations == 0 and result.termination_reason == 'INITIALIZATION_FAILED'


def test_rcig_experiment_uses_in_run_train_results_and_no_train_time_limit(monkeypatch):
    import json, run_rc_ig as entry
    rows = [dict(layout_id=name, seed='20261110', status='FEASIBLE', cmax_s=str(score),runtime_s='1000',
        diagnostic=json.dumps(dict(accounting=dict(search_complete=True),
            initial_saeax=dict(status='FEASIBLE',cmax_s=100.,runtime_s=10.))))
        for name,score in zip(entry.PILOTS,(90.,95.,100.))]
    monkeypatch.setattr(entry,'method_rows',lambda stage:rows)
    assert entry.pilot_decision()['passed']
    rows[0]['cmax_s'] = '100'
    assert not entry.pilot_decision()['passed']
    def forbidden(*args,**kwargs):raise AssertionError('dataset accessed after failed TRAIN')
    monkeypatch.setattr(entry,'load_layouts',forbidden)
    with pytest.raises(ValueError,match='TRAIN screening'):
        entry.run_experiment('validation',1)


def test_rcig_shared_csv_reader_handles_full_repair_traces():
    import csv, io
    from run_saeax import read_rows
    class MemoryCSV:
        def exists(self):return True
        def open(self,**kwargs):return io.StringIO('diagnostic\n'+('x'*200000)+'\n')
    previous = csv.field_size_limit()
    try:
        csv.field_size_limit(131072)
        assert read_rows(MemoryCSV()) == [dict(diagnostic='x'*200000)]
    finally:
        csv.field_size_limit(previous)


def test_crf_dp_all_subset_costs_and_partition_match_small_brute_force():
    """Independent exhaustive enumeration of the SAME fixed-gap neighborhood."""
    from itertools import permutations, product
    import numpy as np
    from mrta_search.crf_dp import _subset_routes, _partition_subsets
    rng = np.random.default_rng(24)
    nodes = np.arange(4, 8, dtype=np.int64)
    parent = np.array([5, 1, 2, 3, 4, 5, 4, 6, 4, 9, 10, 11], np.int64)
    processing = rng.uniform(1., 7., 12)
    travel = rng.uniform(0., 10., (12, 12, 2, 2))
    eligible = np.ones((12, 4), np.bool_)
    eligible[5] = [False, True, False, False]  # Workspace-style fixed X owner.
    eligible[7] = [False, False, True, True]
    expected = np.full((4, 16), np.inf)
    tables = []
    for r in range(4):
        route, bits = np.array([r, 8+r]), np.array([r % 2, (r+1) % 2])
        base = sum(processing[route]) + travel[route[0], route[1], bits[0], bits[1]]
        table = _subset_routes(nodes, r, route, bits, np.array([1, 2]), float(base),
            travel, processing, eligible, parent)
        tables.append(table[0])
        for mask in range(16):
            subset = [int(nodes[i]) for i in range(4) if mask & (1 << i)]
            if any(not eligible[n, r] for n in subset):
                continue
            for order in permutations(subset):
                for vector in product((0, 1), repeat=len(subset)):
                    for position in (1, 2):
                        rr = list(route[:position]) + list(order) + list(route[position:])
                        dd = list(bits[:position]) + list(vector) + list(bits[position:])
                        if any(parent[a] == parent[b] for a, b in zip(rr, rr[1:])):
                            continue
                        value = sum(processing[n] for n in rr) + sum(
                            travel[a, b, x, y] for a, b, x, y in zip(rr, rr[1:], dd, dd[1:]))
                        expected[r, mask] = min(expected[r, mask], value)
        np.testing.assert_allclose(table[0], expected[r], rtol=1.e-13, atol=1.e-12)
    optimum, allocation = _partition_subsets(np.array(tables))
    exhaustive = np.inf
    for owners in product(range(4), repeat=4):
        masks = [sum(1 << i for i, owner in enumerate(owners) if owner == r) for r in range(4)]
        exhaustive = min(exhaustive, max(expected[r, masks[r]] for r in range(4)))
    assert optimum == pytest.approx(exhaustive, abs=1.e-12)
    assert sum(int(m).bit_count() for m in allocation) == 4
    assert int(allocation[0] | allocation[1] | allocation[2] | allocation[3]) == 15
    assert allocation[1] & 2  # Fixed child 5 is assigned to robot 1.


def test_crf_traceback_preserves_modes_heads_and_unmodified_tasks():
    import numpy as np
    from mrta_search import crf_dp as c
    _, w, item, _ = _saeax_fixture()
    original = c.complete_identity(item.routes, item.directions, item.selected)
    candidate = c.reconstruct_fragment(item, w, 0, 1, 3)
    assert candidate is not None
    rr, dd, cc, pp, key, trace = candidate
    c.validate_individual(w, rr, dd, item.selected)
    removed = set(int(n) for n in item.routes[0][1:])
    for r in range(4):
        before = [(int(n), int(d)) for n, d in zip(item.routes[r], item.directions[r]) if int(n) not in removed]
        after = [(int(n), int(d)) for n, d in zip(rr[r], dd[r]) if int(n) not in removed]
        assert before == after
        if len(item.routes[r]):
            assert (rr[r][0], dd[r][0]) == (item.routes[r][0], item.directions[r][0])
        else:
            assert len(rr[r]) == 0  # Explicit head-preserving restriction.
    assert c.complete_identity(item.routes, item.directions, item.selected) == original
    assert trace['lower_bound_s'] == pytest.approx(max(cc))
    assert key[0] == original[0]
    solution = w.solution(rr, item.selected)
    assert tuple(r.block_ids for r in solution.routes) == tuple(tuple(w.ids[n] for n in row) for row in rr)
    with pytest.raises(ValueError, match='head'):
        c.reconstruct_fragment(item, w, 0, 0, 2)


def _crf_mock_search(monkeypatch, *, native=800., repeat=True):
    import numpy as np
    from mrta_search import crf_dp as c
    _, w, item, _, initial = _bc_initial_fixture()
    calls = dict(seed=0, dp=0, native=0, final=0, critical=[])
    def seed(*args, **kwargs):
        calls['seed'] += 1
        return initial
    def windows(current, workspace, settings):
        critical = int(np.argmax(current.costs))
        calls['critical'].append(critical)
        return critical, [(10., 1, 5)] * 6
    def reconstruction(*args, **kwargs):
        index = calls['dp'] % 6 + 1 if repeat else calls['dp']+1
        calls['dp'] += 1
        rr, dd = [row.copy() for row in item.routes], [row.copy() for row in item.directions]
        for bit, (r, k) in enumerate((r, k) for r in (0, 1) for k in range(1, len(dd[r]))):
            dd[r][k] ^= (index >> bit) & 1
        cc, pp = np.array([100., 110., 0., 0.]), np.array([90., 100., 0., 0.])
        return rr, dd, cc, pp, c.complete_identity(rr, dd, item.selected), dict(
            cross_robot_transfers=[], lower_bound_s=110., route_periods_s=cc.tolist())
    def score(*args):
        calls['native'] += 1
        return native-calls['native'] if np.isfinite(native) else native, None, [], None
    monkeypatch.setattr(c, 'run_saeax', seed)
    monkeypatch.setattr(c, 'select_windows', windows)
    monkeypatch.setattr(c, 'reconstruct_fragment', reconstruction)
    monkeypatch.setattr(c, '_priority_arrays', score)
    return c, w, initial, calls


@pytest.mark.parametrize('status,cmax,reason', [
    ('FEASIBLE', 850., 'IMPROVED_B32'), ('FEASIBLE', 1100., 'INITIAL_B32_BETTER_OR_EQUAL'),
    ('DEADLOCK', None, 'FINAL_B32_FAILED')])
def test_crf_best_improvement_one_final_b32_and_cached_initial_fallback(monkeypatch, status, cmax, reason):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    c, w, initial, calls = _crf_mock_search(monkeypatch)
    def final(*args, orientations, **kwargs):
        calls['final'] += 1
        return ScheduleResult(ScheduleStatus(status), cmax=cmax,
            directions=tuple(orientations[r] for r in range(4)))
    monkeypatch.setattr(c, 'reference_schedule_formal', final)
    result = c.run_crf_dp(w.parents)
    a = result.accounting
    assert (calls['seed'], calls['dp'], calls['native'], calls['final']) == (1, 12, 6, 1)
    assert a['duplicate_candidates'] == 6 and a['accepted_improvements'] == 1
    assert a['native_calls'] == 18 and a['b32_calls'] == 4
    assert a['return_reason'] == reason and a['search_complete']
    assert result.rounds == 2 and result.termination_reason == 'NO_IMPROVEMENT'
    assert calls['critical'] == [0, 1]  # Recomputed after the accepted best state.
    assert result.native_cmax == 794. if reason == 'IMPROVED_B32' else result.schedule is initial.schedule
    if status == 'DEADLOCK':
        assert result.final_candidate['cmax_s'] is None


@pytest.mark.parametrize('native', [float('inf'), float('nan'), 950.])
def test_crf_no_native_improvement_has_no_extra_b32(monkeypatch, native):
    c, w, initial, calls = _crf_mock_search(monkeypatch, native=native)
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected final B32')
    monkeypatch.setattr(c, 'reference_schedule_formal', forbidden)
    result = c.run_crf_dp(w.parents)
    assert result.schedule is initial.schedule and result.cmax == initial.cmax
    assert result.accounting['final_b32_calls'] == 0
    assert result.accounting['native_nonfinite'] == (6 if native != 950. else 0)
    assert result.accounting['return_reason'] == 'NO_NATIVE_IMPROVEMENT'


def test_crf_native_budget_and_defaults_not_wall_clock(monkeypatch):
    from types import SimpleNamespace
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    c, w, initial, calls = _crf_mock_search(monkeypatch, repeat=False)
    clock = [0.]
    def elapsed():
        clock[0] += 10.
        return clock[0]
    monkeypatch.setattr(c, 'time', SimpleNamespace(perf_counter=elapsed))
    monkeypatch.setattr(c, 'reference_schedule_formal', lambda *args, orientations, **kwargs:
        ScheduleResult(ScheduleStatus.FEASIBLE, cmax=850., directions=tuple(orientations[r] for r in range(4))))
    result = c.run_crf_dp(w.parents)
    assert result.termination_reason == 'NATIVE_LIMIT' and result.rounds == 7
    assert calls['native'] == 40 and result.runtime > 103.59
    assert result.accounting['accepted_improvements'] == 7
    for kwargs in (dict(max_rounds=9), dict(max_native_evaluations=41), dict(max_fragment=9)):
        with pytest.raises(ValueError):
            c.CRFDPConfig(**kwargs)


def test_crf_lower_bound_prunes_before_native_and_initial_failure_stops(monkeypatch):
    import numpy as np
    c, w, initial, calls = _crf_mock_search(monkeypatch)
    original = c.reconstruct_fragment
    def bounded(*args, **kwargs):
        rr, dd, _, pp, key, trace = original(*args, **kwargs)
        return rr, dd, np.array([900., 110., 0., 0.]), pp, key, trace
    monkeypatch.setattr(c, 'reconstruct_fragment', bounded)
    result = c.run_crf_dp(w.parents)
    assert result.accounting['lower_bound_pruned'] == 6
    assert calls['native'] == 0 and result.accounting['b32_calls'] == 3
    initial.status, initial.cmax = 'DEADLOCK', None
    result = c.run_crf_dp(w.parents)
    assert result.termination_reason == 'INITIALIZATION_FAILED' and result.cmax is None
    assert calls['dp'] == 6  # The failed fresh initialization starts no DP.


def test_crf_pilot_requires_strongest_history_and_time_before_validation(monkeypatch):
    import json
    import run_crf_dp as entry
    rows = [dict(layout_id=name, seed='20261110', status='FEASIBLE', cmax_s=str(score), runtime_s='100',
        diagnostic=json.dumps(dict(accounting=dict(search_complete=True))))
        for name, score in zip(entry.PILOTS, (90., 95., 100.))]
    strong = {(r['layout_id'], int(r['seed'])): dict(r, cmax_s='101') for r in rows}
    monkeypatch.setattr(entry, 'method_rows', lambda stage: rows)
    monkeypatch.setattr(entry, 'pilot_references', lambda: ({}, strong))
    assert entry.pilot_decision()['passed']
    rows[0]['runtime_s'], rows[1]['runtime_s'] = '104', '104'
    assert not entry.pilot_decision()['passed']
    rows[0]['runtime_s'], rows[1]['runtime_s'] = '100', '100'
    rows[0]['cmax_s'], rows[1]['cmax_s'] = '102', '102'
    assert not entry.pilot_decision()['passed']
    def forbidden(*args, **kwargs):
        raise AssertionError('VALIDATION data accessed after failed TRAIN')
    monkeypatch.setattr(entry, 'load_layouts', forbidden)
    with pytest.raises(ValueError, match='TRAIN failed'):
        entry.run_experiment('validation')


def _rcg_fixture():
    import numpy as np
    from mrta_search import gnn_route_columns as c
    parents = tuple(ParentWeld(name, (x, y), (x+.6, y)) for name, x, y in
        (('a', 1., 9.), ('b', 5., 9.), ('c', 9., 9.), ('d', 13., 9.),
         ('e', 2., 1.), ('f', 12., 1.)))
    w = c._HGAWorkspace(parents, ScientificConfig(), FORMAL_SCOPE_V2)
    ids = {w.ids[n]: int(n) for m in w.initial_selected for n in w.pattern_nodes[m] if n >= 0}
    def bundle(names):
        routes = [[ids[name+'::whole'] for name in row] for row in names]
        return c.make_bundle(w, routes, [[0]*len(row) for row in routes], w.initial_selected)
    initial = bundle((('a', 'b', 'c'), ('d',), ('e',), ('f',)))
    balanced = bundle((('a', 'b'), ('c', 'd'), ('e',), ('f',)))
    return c, w, initial, balanced


def test_rcg_joint_integer_master_matches_exhaustive_route_combinations():
    import itertools
    c, w, initial, balanced = _rcg_fixture()
    pools = [[a] if a.key == b.key else [a, b] for a, b in zip(initial.columns, balanced.columns)]
    master = c.RestrictedMaster(w, pools)
    lp, dual = master.solve_lp()
    mip, chosen = master.solve_integer(2.)
    feasible = []
    for columns in itertools.product(*pools):
        try:
            b = c.make_bundle(w, [x.nodes for x in columns], [x.directions for x in columns], initial.selected)
        except ValueError:
            continue
        feasible.append(b)
    assert mip.success and chosen is not None and lp.fun <= chosen.period+1.e-7
    assert chosen.period == pytest.approx(min(b.period for b in feasible))
    assert chosen.period < initial.period and chosen.key == balanced.key
    assert len(chosen.columns[0].nodes) == len(chosen.columns[1].nodes) == 2
    solution, directions = c.export_bundle(w, chosen)
    assert tuple(r.block_ids for r in solution.routes) == tuple(tuple(w.ids[n] for n in x.nodes) for x in chosen.columns)


def test_rcg_lp_dual_signs_and_new_column_price_are_actual_matrix_marginals():
    import numpy as np
    c, w, initial, balanced = _rcg_fixture()
    old = c.RestrictedMaster(w, [[x] for x in initial.columns])
    lp, dual = old.solve_lp()
    assert sum(dual.load) == pytest.approx(1.) and min(dual.load) >= -1.e-8
    assert old.existing_reduced_costs(lp) == pytest.approx(lp.lower.marginals[:old.nx])
    assert np.max(np.abs(old.existing_reduced_costs(lp))) < 1.e-7
    enlarged = c.RestrictedMaster(w, [[a, b] if a.key != b.key else [a]
        for a, b in zip(initial.columns, balanced.columns)])
    extended_mu = np.r_[lp.ineqlin.marginals[:4], np.zeros(len(enlarged.conflicts))]
    for i, column in enumerate(enlarged.columns):
        if column.key not in {x.key for x in initial.columns}:
            actual = (-enlarged.aeq[:, i].T@lp.eqlin.marginals-enlarged.aub[:, i].T@extended_mu).item()
            assert dual.reduced_cost(column) == pytest.approx(actual)
    assert dual.reduced_cost(balanced.columns[0]) < 0
    assert dual.reduced_cost(balanced.columns[1]) >= 0  # necessary coverage companion


def test_rcg_master_excludes_incompatible_different_first_tasks():
    c, w, initial, balanced = _rcg_fixture()
    # Reversing the first two heads violates the original same-rail ordering.
    left = c.make_column(w, 0, initial.columns[0].nodes[::-1], (0, 0, 0))
    right = c.make_column(w, 1, (initial.columns[0].nodes[0],), (0,))
    pools = [[initial.columns[0], left], [initial.columns[1], right],
        [initial.columns[2]], [initial.columns[3]]]
    master = c.RestrictedMaster(w, pools)
    a = master.columns.index(left); b = master.columns.index(right)
    assert (a, b) in master.conflicts
    assert not c.heads_compatible(w, left, right)
    result, chosen = master.solve_integer(2.)
    assert result.success
    c.make_bundle(w, [x.nodes for x in chosen.columns], [x.directions for x in chosen.columns], chosen.selected)


def test_rcg_master_alternative_modes_cover_all_children_and_fixed_x_owners():
    import itertools
    from mrta_baselines.hga import _repair_oriented_first
    from mrta_search import gnn_route_columns as c
    _, w, a, _ = _eax_mixed_fixture()
    rr, dd = _repair_oriented_first(a.routes, a.directions, w.points, w.travel, w.parameters, len(w.ids))
    split = c.make_bundle(w, rr, dd, a.selected)
    whole = c._training_seed(w)
    xp = next(p for p, parent in enumerate(w.parents) if parent.parent_id == 'x')
    assert w.patterns[split.selected[xp]].kind is SplitKind.X_SPLIT
    assert w.patterns[whole.selected[xp]].kind is SplitKind.WHOLE
    for required in (split, whole):
        _, b = c.RestrictedMaster(w, [[x] for x in required.columns]).solve_integer(2.)
        assert tuple(b.selected) == tuple(required.selected)
    xnodes = [int(n) for n in w.pattern_nodes[split.selected[xp]] if n >= 0]
    for node in xnodes:
        owner = next(x.robot for x in split.columns if node in x.nodes)
        assert w.eligible[node].sum() == 1 and w.eligible[node, owner]
        with pytest.raises(ValueError, match='qualification'):
            c.make_column(w, owner ^ 1, (node,), (0,))
    with pytest.raises(ValueError, match='mutually exclusive'):
        c.make_column(w, 0, (xnodes[0], int(w.pattern_nodes[whole.selected[xp], 0])), (0, 0))
    pools = [[a] if a.key == b.key else [a, b] for a, b in zip(split.columns, whole.columns)]
    master = c.RestrictedMaster(w, pools)
    _, chosen = master.solve_integer(2.)
    expected = []
    for columns in itertools.product(*pools):
        for modes in (split.selected, whole.selected):
            try:
                expected.append(c.make_bundle(w, [x.nodes for x in columns], [x.directions for x in columns], modes))
            except ValueError:
                pass
    assert chosen.period == pytest.approx(min(x.period for x in expected))


def test_rcg_columns_reject_duplicate_missing_children_and_canonical_siblings():
    c, w, initial, _ = _rcg_fixture()
    node = initial.columns[0].nodes[0]
    with pytest.raises(ValueError, match='duplicate'):
        c.make_column(w, 0, (node, node), (0, 1))
    with pytest.raises(ValueError, match='unaligned'):
        c.make_column(w, 0, (node,), ())
    with pytest.raises(ValueError, match='missing'):
        c.make_bundle(w, [x.nodes[1:] for x in initial.columns],
            [x.directions[1:] for x in initial.columns], initial.selected)


def test_rcg_gnn_scores_append_choices_and_changes_columns_without_local_operators(monkeypatch):
    import torch
    c, w, initial, _ = _rcg_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('pricing called a scheduler or old search')
    monkeypatch.setattr(c, 'reference_schedule_formal', forbidden)
    monkeypatch.setattr(c, 'run_saeax', forbidden)
    _, dual = c.RestrictedMaster(w, [[x] for x in initial.columns]).solve_lp()
    torch.manual_seed(7)
    policy = c.RoutePricingGNN()
    # Controlled head: actively reverse the actual reduced-cost ordering.
    monkeypatch.setattr(policy, 'forward', lambda rows: -rows[:, -c.FEATURE_DIM+4])
    context = c.PricingContext(w, policy)
    counts = dict(gnn_score_batches=0, gnn_scored_actions=0, gnn_top_choice_differences=0)
    guided = c.construct_bundle(context, dual, initial, 0, counts=counts)
    greedy = c.construct_bundle(c.PricingContext(w, None), dual, initial, 0)
    assert guided is not None and greedy is not None and guided.key != greedy.key
    assert counts['gnn_scored_actions'] > 0 and counts['gnn_top_choice_differences'] > 0


def test_rcg_new_pricing_head_trains_from_lp_completions_with_frozen_encoder(monkeypatch):
    import torch
    c, w, initial, _ = _rcg_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('pricing labels used B32 or an old optimizer')
    monkeypatch.setattr(c, 'reference_schedule_formal', forbidden)
    monkeypatch.setattr(c, 'run_saeax', forbidden)
    monkeypatch.setattr(c, '_training_seed', lambda workspace: initial)
    torch.manual_seed(11); torch.set_num_threads(1)
    policy = c.RoutePricingGNN()
    old = {k: v.detach().clone() for k, v in policy.encoder.state_dict().items()}
    head = policy.head[0].weight.detach().clone()
    fit = c.fit_pricing_head(policy, [('toy_train', w.parents)], epochs=2)
    assert fit['b32_calls'] == fit['native_calls'] == 0 and fit['lp_calls'] == 3
    assert fit['complete_route_labels'] > 0 and fit['training_rank_pairs'] > 0
    assert fit['final_head_gradient_norm'] > 0
    assert not torch.equal(head[:, :64], policy.head[0].weight[:, :64])
    assert all(torch.equal(value, policy.encoder.state_dict()[key]) for key, value in old.items())


@pytest.mark.parametrize('status,score,improves', [('FEASIBLE', 1., True),
    ('FEASIBLE', 10000., False), ('DEADLOCK', None, False)])
def test_rcg_final_b32_only_and_cached_same_run_initial_fallback(monkeypatch, status, score, improves):
    from types import SimpleNamespace
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    c, w, original, balanced = _rcg_fixture()
    solution, directions = c.export_bundle(w, original)
    initial = SimpleNamespace(status='FEASIBLE', cmax=original.period+100., solution=solution,
        directions=directions, schedule=ScheduleResult(ScheduleStatus.FEASIBLE,
        cmax=original.period+100., directions=directions), runtime=3.,
        native_cmax=original.period, accounting=dict(native_calls=12, b32_calls=3))
    calls = []
    monkeypatch.setattr(c, 'construct_bundle', lambda *args, **kwargs: balanced)
    def final(solution, config, *, scope, orientations):
        calls.append(solution)
        return ScheduleResult(ScheduleStatus(status), cmax=score,
            directions=tuple(orientations[r] for r in range(4)))
    monkeypatch.setattr(c, 'reference_schedule_formal', final)
    monkeypatch.setattr(c, 'run_saeax', lambda *a, **k: pytest.fail('initialization repeated'))
    result = c.optimize_route_columns(w.parents, initial, settings=c.RCGConfig(max_rounds=3, mip_seconds=1.))
    assert len(calls) == 1 and result.accounting['final_b32_calls'] == 1
    assert result.accounting['b32_calls'] == 4 and result.accounting['native_calls'] == 12
    assert result.accounting['search_complete'] and result.runtime >= 3.
    assert result.accounting['negative_columns'] > 0 and result.accounting['companion_columns'] > 0
    if improves:
        assert result.cmax == 1. and result.schedule is not initial.schedule
        assert result.accounting['changes']
    else:
        assert result.schedule is initial.schedule and result.cmax == initial.cmax
    if status == 'DEADLOCK':
        assert result.final_candidates[0]['cmax_s'] is None


def test_rcg_two_empty_rounds_stop_and_do_not_reevaluate_initial(monkeypatch):
    from types import SimpleNamespace
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    c, w, original, _ = _rcg_fixture()
    solution, directions = c.export_bundle(w, original)
    initial = SimpleNamespace(status='FEASIBLE', cmax=original.period, solution=solution,
        directions=directions, schedule=ScheduleResult(ScheduleStatus.FEASIBLE,
        cmax=original.period, directions=directions), runtime=2.,
        accounting=dict(native_calls=10, b32_calls=1))
    monkeypatch.setattr(c, 'construct_bundle', lambda *a, **k: original)
    monkeypatch.setattr(c, 'reference_schedule_formal', lambda *a, **k: pytest.fail('duplicate B32'))
    result = c.optimize_route_columns(w.parents, initial)
    assert result.accounting['rounds'] == 2 and result.accounting['lp_calls'] == 2
    assert result.accounting['milp_calls'] == 0 and result.accounting['final_b32_calls'] == 0
    assert result.schedule is initial.schedule
    assert result.termination_reason == 'NO_NEW_VALUABLE_COLUMNS'


def test_rcg_train_requires_initial_gain_gnn_ablation_and_runtime_before_validation(monkeypatch):
    import json
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_gnn_route_columns as entry
    def records(scores):
        return [dict(layout_id=name, seed='20261110', status='FEASIBLE', cmax_s=str(score),
            runtime_s='50', diagnostic=json.dumps(dict(accounting=dict(search_complete=True))))
            for name, score in zip(entry.PILOTS, scores)]
    data = dict(gnn=records((99., 99., 100.)), initial=records((100., 100., 100.)),
        greedy=records((99.5, 99.5, 100.)))
    monkeypatch.setattr(entry, 'method_rows', lambda stage, method='gnn': data[method])
    assert entry.pilot_decision()['passed']
    data['greedy'] = records((98., 98., 100.))
    assert not entry.pilot_decision()['passed']
    data['greedy'] = records((99.5, 99.5, 100.))
    data['gnn'] = records((99.9, 99.9, 100.))
    assert not entry.pilot_decision()['passed']  # below 0.5% median improvement
    monkeypatch.setattr(entry, 'load_layouts', lambda *a: pytest.fail('read VALIDATION after failed TRAIN'))
    with pytest.raises(ValueError, match='TRAIN failed'):
        entry.run_experiment('validation')

