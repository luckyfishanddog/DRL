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


def test_common_model_bounded_hga_defaults_preserve_paper_parameters():
    config = AdaptedHGAConfig()
    assert (config.mu, config.lambda_size, config.alpha, config.max_iterations) == (
        20,
        10,
        20,
        200000,
    )
    assert config.initialization_reference_limit == config.mu
    assert config.vnd_candidate_cap == 256
