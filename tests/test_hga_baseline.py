from __future__ import annotations

import random

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


def test_oriented_vnd_exhausts_all_six_fixed_gene_neighborhoods():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    config, _, solution = _fixture()
    w = h._HGAWorkspace(solution.parents, config, None, patterns=solution.patterns)
    routes = w.encode(solution)
    bits = [np.arange(len(r), dtype=np.int64) % 2 for r in routes]
    result = h._vnd_oriented_arrays(routes, bits, w.travel, w.processing, w.points, w.eligible, w.parent_index,
                                    w.nearest, w.parameters, 20, 0, 0, w.oriented_scratch, np.zeros(1, np.int64))
    current, directions, score, operations, _, pass_hit, candidate_hit, _ = result
    assert np.isfinite(score) and operations > 0 and not pass_hit and not candidate_hit
    assert score == _priority_arrays(current, w.travel, w.processing, w.points, w.parameters, directions)[0]
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
                        value = _priority_arrays(candidate, w.travel, w.processing, w.points, w.parameters, db)[0]
                        assert not np.isfinite(value) or value >= score - 1.e-8


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


def test_oriented_prefix_bounds_match_complete_native_search_with_waits():
    import numpy as np
    from mrta_baselines import hga as h
    from mrta_baselines.wag_vns import _priority_arrays
    w, solution = _oriented_wait_fixture()
    routes = w.encode(solution)
    _, bits, original_ops, _ = _priority_arrays(routes, w.travel, w.processing, w.points, w.parameters)
    assert any(op[8] == 4. for group in original_ops for op in group)
    args = (routes, bits, w.travel, w.processing, w.points, w.eligible, w.parent_index, w.nearest,
            w.parameters, 20, 0, 0, w.oriented_scratch)
    full = h._vnd_oriented_arrays(*args, np.zeros(3, np.int64), False)
    hits = np.zeros(3, np.int64)
    bounded = h._vnd_oriented_arrays(*args, hits, True)
    for actual, expected in zip(bounded[0] + bounded[1], full[0] + full[1]):
        np.testing.assert_array_equal(actual, expected)
    assert bounded[2:] == full[2:]

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

