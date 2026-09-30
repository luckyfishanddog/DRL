from __future__ import annotations

import random

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

