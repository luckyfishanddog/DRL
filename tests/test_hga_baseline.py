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
    from mrta_baselines.wag_vns import _interval_costs, _array_route_cost
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
    score = h.hga_surrogate(result, config)[0]
    workspace = h._HGAWorkspace(result.parents, config, None, patterns=result.patterns)
    for move in range(1, 7):
        for route in result.routes:
            for u in route.block_ids:
                for v in h.alpha_nearest(result, u, 20, config):
                    candidate = h.apply_hga_neighborhood(result, move, u, v, config)
                    if candidate is not None and h._legal_first_arrays(workspace.encode(candidate), workspace.points, workspace.parameters):
                        assert h.hga_surrogate(candidate, config)[0] >= score - 1e-8


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
    _assert_valid(result.solution, ScientificConfig())
