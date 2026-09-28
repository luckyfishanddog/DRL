from __future__ import annotations


def test_formal_scope_hash_separates_f1_f4_from_numeric_config():
    from dataclasses import FrozenInstanceError, replace
    import hashlib
    import pytest
    from mrta_reference.model import FORMAL_SCOPE_V1, FormalScope, ScientificConfig, RunScientificIdentity
    scope = FORMAL_SCOPE_V1
    assert scope.scope_hash == FormalScope().scope_hash
    assert scope.scope_hash == hashlib.sha256(scope.canonical_json.encode()).hexdigest()
    for change in (
        {"optional_x_split_policy": "INCLUDED"}, {"pattern_domain": ("WHOLE",)},
        {"terminal_policy": "OCCUPY_TO_CMAX"}, {"empty_route_policy": "PARK"},
        {"initial_deployment_policy": "PAID"}, {"deadlock_policy_id": "OTHER"},
        {"deadlock_state_budget": 32}, {"reference_scheduler_policy_id": "OTHER"},
    ):
        mutated = replace(scope, **change)
        assert mutated.scope_hash != scope.scope_hash
        with pytest.raises(ValueError, match="unsupported"):
            mutated.validate_implemented()
    with pytest.raises(FrozenInstanceError):
        scope.deadlock_state_budget = 32
    first = RunScientificIdentity.from_scope(scope, ScientificConfig(), "commit-a")
    second = RunScientificIdentity.from_scope(scope, ScientificConfig(weld_speed=0.02), "commit-b")
    assert first.scope_hash == second.scope_hash
    assert first.scientific_config_hash != second.scientific_config_hash

import itertools
import math
import random
from dataclasses import replace

import pytest

from mrta_reference.geometry import (
    blocks_for_pattern,
    frozen_handover_centers,
    generate_x_split_patterns,
    generate_y_split_patterns,
    optimize_directions,
    whole_eligible_rails,
    x_split_relation,
)
from mrta_reference.model import (
    ParentWeld,
    Rail,
    ScientificAmbiguityError,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)


CONFIG = ScientificConfig()


def test_scientific_defaults_and_whole_eligibility() -> None:
    assert CONFIG.workspace_x == (0.0, 20.0)
    assert CONFIG.workspace_y == (0.0, 12.0)
    assert CONFIG.weld_speed == 0.0108
    assert CONFIG.empty_speed == 0.20
    assert CONFIG.t_pre == 20.0
    assert CONFIG.t_post == 30.0
    assert CONFIG.by == (5.8, 6.2)
    assert whole_eligible_rails((0, 6.0), (2, 6.1), CONFIG) == frozenset((Rail.UPPER, Rail.LOWER))
    assert whole_eligible_rails((0, 7.0), (2, 8.0), CONFIG) == frozenset((Rail.UPPER,))
    assert whole_eligible_rails((0, 4.0), (2, 5.0), CONFIG) == frozenset((Rail.LOWER,))
    assert whole_eligible_rails((0, 5.0), (2, 7.0), CONFIG) == frozenset()


def test_scientific_config_serialization_and_hash_are_value_only_and_stable() -> None:
    same = ScientificConfig()
    changed = replace(CONFIG, t_pre=CONFIG.t_pre + 1.0)
    assert same.canonical_json == CONFIG.canonical_json
    assert same.scientific_hash == CONFIG.scientific_hash
    assert changed.scientific_hash != CONFIG.scientific_hash
    assert len(CONFIG.scientific_hash) == 64
    assert "timestamp" not in CONFIG.canonical_json
    assert "machine" not in CONFIG.canonical_json


def test_y_candidates_are_only_boundaries_center_midpoint_and_deterministically_deduplicated() -> None:
    parent = ParentWeld("cross", (1.0, 5.0), (1.0, 7.0))
    candidates = generate_y_split_patterns(parent, CONFIG)
    assert [candidate.point_id for candidate in candidates] == ["BY_LOWER", "BY_CENTER", "BY_UPPER"]
    assert [candidate.t for candidate in candidates] == pytest.approx([0.4, 0.5, 0.6])
    assert all(candidate.mandatory for candidate in candidates)
    assert candidates == generate_y_split_patterns(parent, CONFIG)
    # The geometric midpoint duplicates BY_CENTER here and is deterministically removed.
    assert "MIDPOINT" not in {candidate.point_id for candidate in candidates}

    with pytest.raises(ValueError, match="deterministic legal candidates"):
        blocks_for_pattern(
            parent,
            SplitPattern("cross", SplitKind.Y_SPLIT, 0.5, "wrong-id"),
            CONFIG,
        )


def test_split_length_conservation_and_lmin_validation() -> None:
    parent = ParentWeld("p", (0.0, 8.0), (1.0, 8.0))
    pattern = SplitPattern("p", SplitKind.X_SPLIT, 0.2, "given-x")
    blocks = blocks_for_pattern(parent, pattern, CONFIG)
    assert len(blocks) == 2
    assert sum(block.length for block in blocks) == pytest.approx(parent.length)
    assert min(block.length for block in blocks) >= CONFIG.min_child_length
    with pytest.raises(ValueError, match="Lmin"):
        blocks_for_pattern(parent, SplitPattern("p", SplitKind.X_SPLIT, 0.19, "short"), CONFIG)

    exactly = blocks_for_pattern(
        parent, SplitPattern("p", SplitKind.X_SPLIT, 0.2, "exact-lmin"), CONFIG
    )
    assert exactly[0].length == pytest.approx(CONFIG.min_child_length)
    with pytest.raises(ValueError, match="Lmin"):
        blocks_for_pattern(
            parent,
            SplitPattern(
                "p",
                SplitKind.X_SPLIT,
                CONFIG.min_child_length - 2.0 * CONFIG.numeric_epsilon,
                "just-below-lmin",
            ),
            CONFIG,
        )


def test_zero_and_near_zero_parent_lengths_are_handled_numerically() -> None:
    with pytest.raises(ValueError, match="positive length"):
        ParentWeld("zero", (1.0, 1.0), (1.0, 1.0))
    near_zero = ParentWeld("near-zero", (1.0, 1.0), (1.0 + 1.0e-12, 1.0))
    with pytest.raises(ValueError, match="Lmin"):
        blocks_for_pattern(
            near_zero,
            SplitPattern("near-zero", SplitKind.X_SPLIT, 0.5, "mid"),
            CONFIG,
        )


def test_x_split_given_point_relation_without_invented_candidate_enumeration() -> None:
    parent = ParentWeld("x", (0.0, 8.0), (4.0, 8.0))
    pattern = SplitPattern("x", SplitKind.X_SPLIT, 0.75, "caller-supplied")
    blocks = blocks_for_pattern(parent, pattern, CONFIG)
    assert blocks[0].end == (3.0, 8.0)
    assert x_split_relation(blocks[0].end, 3.1, CONFIG) == "IN_BX"
    assert x_split_relation((2.0, 8.0), 3.1, CONFIG) == "LEFT_OF_BX"
    with pytest.raises(ScientificAmbiguityError, match="not frozen"):
        generate_x_split_patterns(parent, 3.1, CONFIG)
    supplied = generate_x_split_patterns(
        parent,
        3.1,
        CONFIG,
        provider=lambda *_: ((0.75, "caller-supplied"),),
    )
    assert supplied == (pattern,)


def test_weighted_median_is_leftmost_deterministic_and_clipped() -> None:
    parents = (
        ParentWeld("a", (-2.0, 8.0), (2.0, 8.0)),  # midpoint 0 -> clipped 0.2
        ParentWeld("b", (9.0, 8.0), (11.0, 8.0)),
        ParentWeld("c", (9.0, 4.0), (11.0, 4.0)),
    )
    up, low = frozen_handover_centers(parents, CONFIG)
    assert up == 0.2  # equal weights: the leftmost weighted median wins
    assert low == 10.0
    assert (up, low) == frozen_handover_centers(tuple(reversed(parents)), CONFIG)
    assert frozen_handover_centers((ParentWeld("only-up", (2, 8), (3, 8)),), CONFIG)[1] == 10.0


def _brute_force(blocks: tuple[WeldingBlock, ...]):
    choices = []
    for vector in itertools.product((0, 1), repeat=len(blocks)):
        cost = 0.0
        for index in range(1, len(blocks)):
            previous_end = blocks[index - 1].end if vector[index - 1] == 0 else blocks[index - 1].start
            current_start = blocks[index].start if vector[index] == 0 else blocks[index].end
            cost += math.dist(previous_end, current_start) / CONFIG.empty_speed
        choices.append((cost, vector))
    return min(choices, key=lambda item: (item[0], item[1]))


@pytest.mark.parametrize("count", range(1, 9))
def test_direction_dp_matches_exhaustive_2_to_m(count: int) -> None:
    rng = random.Random(9182 + count)
    blocks = tuple(
        WeldingBlock(
            f"p{index}",
            f"p{index}::whole",
            0.0,
            1.0,
            (rng.uniform(0, 20), rng.uniform(0, 12)),
            (rng.uniform(0, 20), rng.uniform(0, 12)),
        )
        for index in range(count)
    )
    expected_cost, expected_vector = _brute_force(blocks)
    actual = optimize_directions(blocks, CONFIG)
    assert actual.empty_travel_time == pytest.approx(expected_cost, abs=1.0e-12)
    assert actual.orientations == expected_vector


def test_direction_dp_is_open_route_and_tie_breaks_forward() -> None:
    block = WeldingBlock("p", "p::whole", 0.0, 1.0, (19.0, 10.0), (20.0, 10.0))
    result = optimize_directions((block,), CONFIG)
    assert result.empty_travel_time == 0.0
    assert result.orientations == (0,)


def test_direction_dp_exact_ties_reversed_geometry_and_zero_transition() -> None:
    first = WeldingBlock("a", "a::whole", 0.0, 1.0, (0.0, 0.0), (1.0, 0.0))
    second = WeldingBlock("b", "b::whole", 0.0, 1.0, (1.0, 0.0), (0.0, 0.0))
    result = optimize_directions((first, second), CONFIG)
    expected_cost, expected_vector = _brute_force((first, second))
    assert result.empty_travel_time == pytest.approx(0.0)
    assert result.empty_travel_time == pytest.approx(expected_cost)
    assert result.orientations == expected_vector == (0, 0)
