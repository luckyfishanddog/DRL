from __future__ import annotations


def test_formal_scope_hash_separates_f1_f4_from_numeric_config():
    from dataclasses import FrozenInstanceError, replace
    import hashlib
    import pytest
    from mrta_reference.model import (
        ACTIVE_FORMAL_SCOPE,
        FORMAL_SCOPE_V1,
        FORMAL_SCOPE_V1_1,
        FORMAL_SCOPE_V2,
        FormalScope,
        ScientificConfig,
        RunScientificIdentity,
    )
    from mrta_reference.provenance import SourceProvenance
    scope = FORMAL_SCOPE_V1
    assert scope.scope_hash == "8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9"
    assert FORMAL_SCOPE_V1_1.scope_hash == "5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc"
    assert FORMAL_SCOPE_V2.scope_hash == "16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599"
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
    assert ACTIVE_FORMAL_SCOPE is FORMAL_SCOPE_V2
    assert FORMAL_SCOPE_V1_1.scope_hash != FORMAL_SCOPE_V1.scope_hash
    assert FORMAL_SCOPE_V1_1.deadlock_budget_unit == "COMPLETE_ALTERNATIVE_ROLLOUTS"
    assert FORMAL_SCOPE_V1_1.deadlock_rollout_budget == 32
    for field in (
        "pattern_domain",
        "optional_x_split_policy",
        "y_split_rule_id",
        "max_split_per_parent",
        "terminal_policy",
        "empty_route_policy",
        "initial_deployment_policy",
        "open_route_policy_id",
        "objective_policy_id",
        "certifier_policy_id",
        "interference_policy_id",
    ):
        assert getattr(FORMAL_SCOPE_V1_1, field) == getattr(FORMAL_SCOPE_V1, field)
    FORMAL_SCOPE_V1.validate_implemented()
    FORMAL_SCOPE_V1_1.validate_implemented()
    FORMAL_SCOPE_V2.validate_implemented()
    for field in (
        "x_split_rule_id",
        "x_split_assignment_policy_id",
        "x_split_processing_policy_id",
        "x_split_shared_point_policy_id",
        "combined_xy_split_policy_id",
    ):
        assert getattr(FORMAL_SCOPE_V1, field) is None
        assert getattr(FORMAL_SCOPE_V1_1, field) is None
        assert getattr(FORMAL_SCOPE_V2, field) is not None
    first = RunScientificIdentity.from_scope(
        scope, ScientificConfig(),
        SourceProvenance("luckyfishanddog/DRL", "commit-a", "tree-a", False, True),
    )
    second = RunScientificIdentity.from_scope(
        scope, ScientificConfig(weld_speed=0.02),
        SourceProvenance("luckyfishanddog/DRL", "commit-b", "tree-b", True, False),
    )
    assert first.scope_hash == second.scope_hash
    assert first.scientific_config_hash != second.scientific_config_hash
    assert first.repository_id == "luckyfishanddog/DRL"
    assert first.source_tree_hash == "tree-a" and first.commit_verified


def test_v2_shared_catalog_is_deterministic_finite_and_mandatory_y_closed():
    from mrta_reference.geometry import (
        build_legal_pattern_catalog,
        pattern_catalog_hash,
    )
    from mrta_reference.model import (
        FORMAL_SCOPE_V2,
        ParentWeld,
        ScientificConfig,
        SplitKind,
    )

    config = ScientificConfig()
    parents = (
        ParentWeld("short", (1.0, 8.0), (2.0, 8.0)),
        ParentWeld("vertical", (4.0, 8.0), (4.0, 9.0)),
        ParentWeld("mandatory", (3.0, 2.0), (3.5, 10.0)),
    )
    first = build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)
    second = build_legal_pattern_catalog(tuple(reversed(parents)), config, FORMAL_SCOPE_V2)
    assert first == second
    assert pattern_catalog_hash(parents, config, FORMAL_SCOPE_V2) == pattern_catalog_hash(
        tuple(reversed(parents)), config, FORMAL_SCOPE_V2
    )
    short_x = [item for item in first["short"] if item.kind is SplitKind.X_SPLIT]
    assert short_x
    assert {item.point_id for item in short_x} <= {
        "BX_LOWER", "BX_CENTER", "BX_UPPER", "MIDPOINT"
    }
    assert not any(item.kind is SplitKind.X_SPLIT for item in first["vertical"])
    assert first["mandatory"]
    assert all(item.kind is SplitKind.Y_SPLIT for item in first["mandatory"])
    assert all(item.mandatory for item in first["mandatory"])


def _git(cwd, *args):
    import subprocess
    return subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True, check=True
    ).stdout.strip()


def _make_git_project(path, remote):
    (path / "src").mkdir(parents=True)
    (path / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n", encoding="utf-8")
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "fixture")
    _git(path, "remote", "add", "origin", remote)
    return _git(path, "rev-parse", "HEAD")


def test_source_provenance_rejects_nested_outer_git_and_allows_explicit_development(tmp_path):
    import pytest
    from mrta_reference.provenance import SourceProvenanceError, resolve_source_provenance
    outer = tmp_path / "outer"
    project = outer / "DRL"
    commit = _make_git_project(outer, "https://github.com/luckyfishanddog/DRL.git")
    (project / "src").mkdir(parents=True)
    (project / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='nested'\nversion='0'\n", encoding="utf-8")
    with pytest.raises(SourceProvenanceError, match="git root"):
        resolve_source_provenance(project)
    unverified = resolve_source_provenance(
        project, source_commit=commit, allow_unverified_source=True
    )
    assert not unverified.commit_verified
    assert unverified.repository_id == "luckyfishanddog/DRL"
    with pytest.raises(SourceProvenanceError, match="verified"):
        unverified.require_formal_result()


def test_source_provenance_auto_verifies_matching_root_and_rejects_wrong_remote(tmp_path):
    import pytest
    from mrta_reference.provenance import SourceProvenanceError, resolve_source_provenance
    matching = tmp_path / "matching"
    commit = _make_git_project(matching, "git@github.com:luckyfishanddog/DRL.git")
    provenance = resolve_source_provenance(matching)
    assert provenance.commit_verified and not provenance.worktree_dirty
    assert provenance.source_commit == commit
    provenance.require_formal_result()
    wrong = tmp_path / "wrong"
    _make_git_project(wrong, "https://github.com/luckyfishanddog/MRTA.git")
    with pytest.raises(SourceProvenanceError, match="no remote"):
        resolve_source_provenance(wrong)


def test_source_tree_hash_is_stable_sensitive_and_ignores_cache(tmp_path):
    from mrta_reference.provenance import compute_source_tree_hash
    root = tmp_path / "tree"
    (root / "src" / "pkg").mkdir(parents=True)
    source = root / "src" / "pkg" / "module.py"
    source.write_bytes(b"VALUE = 1\n")
    (root / "pyproject.toml").write_bytes(b"[project]\nname='tree'\n")
    first = compute_source_tree_hash(root)
    assert first == compute_source_tree_hash(root)
    cache = root / "src" / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"generated")
    assert compute_source_tree_hash(root) == first
    source.write_bytes(b"VALUE = 2\n")
    assert compute_source_tree_hash(root) != first

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
    x_split_geometry_metadata,
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
    assert CONFIG.by == (5.5, 6.5)
    assert CONFIG.delta_x == CONFIG.delta_y == 0.5
    assert CONFIG.min_child_length == 0.2
    assert CONFIG.interference_dx == CONFIG.interference_dy == 0.5
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
    assert [candidate.point_id for candidate in candidates] == ["BY_OUTER_LOWER", "BY_LOWER", "BY_CENTER", "BY_UPPER", "BY_OUTER_UPPER"]
    assert [candidate.t for candidate in candidates] == pytest.approx([0.25, 0.4, 0.5, 0.6, 0.75])
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
    pattern = SplitPattern("p", SplitKind.X_SPLIT, 0.2, "given-x", rail=Rail.UPPER)
    blocks = blocks_for_pattern(parent, pattern, CONFIG)
    assert len(blocks) == 2
    assert sum(block.length for block in blocks) == pytest.approx(parent.length)
    assert min(block.length for block in blocks) >= CONFIG.min_child_length
    with pytest.raises(ValueError, match="Lmin"):
        blocks_for_pattern(parent, SplitPattern("p", SplitKind.X_SPLIT, 0.19, "short", rail=Rail.UPPER), CONFIG)

    exactly = blocks_for_pattern(
        parent, SplitPattern("p", SplitKind.X_SPLIT, 0.2, "exact-lmin", rail=Rail.UPPER), CONFIG
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
                rail=Rail.UPPER,
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
            SplitPattern("near-zero", SplitKind.X_SPLIT, 0.5, "mid", rail=Rail.LOWER),
            CONFIG,
        )


def test_x_split_given_point_relation_and_explicit_rail_enumeration() -> None:
    parent = ParentWeld("x", (0.0, 8.0), (4.0, 8.0))
    pattern = SplitPattern("x", SplitKind.X_SPLIT, 0.75, "caller-supplied", rail=Rail.UPPER)
    blocks = blocks_for_pattern(parent, pattern, CONFIG)
    assert blocks[0].end == (3.0, 8.0)
    assert x_split_relation(blocks[0].end, 3.1, CONFIG) == "IN_BX"
    assert x_split_relation((2.0, 8.0), 3.1, CONFIG) == "LEFT_OF_BX"
    with pytest.raises(ScientificAmbiguityError, match="explicit rail"):
        generate_x_split_patterns(parent, 3.1, CONFIG)
    supplied = generate_x_split_patterns(
        parent,
        3.1,
        CONFIG,
        provider=lambda *_: ((0.75, "caller-supplied"),),
    )
    assert supplied == (pattern,)


def test_finite_x_candidates_are_deterministic_geometry_only_and_deduplicated() -> None:
    parent = ParentWeld("finite-x", (1.0, 8.0), (5.0, 8.0))
    first = generate_x_split_patterns(parent, 3.0, CONFIG, rail=Rail.UPPER)
    second = generate_x_split_patterns(parent, 3.0, CONFIG, rail=Rail.UPPER)
    assert first == second
    assert [item.point_id for item in first] == ["BX_OUTER_LOWER", "BX_LOWER", "BX_CENTER", "BX_UPPER", "BX_OUTER_UPPER"]
    assert [item.t for item in first] == pytest.approx([0.375, 0.45, 0.5, 0.55, 0.625])
    assert all(item.rail is Rail.UPPER for item in first)
    assert all(parent.point(item.t)[0] != 10.0 for item in first if item.t is not None)


def test_x_candidate_filter_has_no_wait_input_and_rejects_no_span_and_short_children() -> None:
    vertical = ParentWeld("vertical", (2.0, 7.0), (2.0, 9.0))
    assert generate_x_split_patterns(vertical, 2.0, CONFIG, rail=Rail.UPPER) == ()
    short = ParentWeld("short", (0.0, 8.0), (0.3, 8.0))
    assert generate_x_split_patterns(short, 0.15, CONFIG, rail=Rail.UPPER) == ()


def test_x_blocks_use_spatial_left_right_identity_for_reversed_parent() -> None:
    parent = ParentWeld("reverse", (5.0, 8.0), (1.0, 8.0))
    pattern = generate_x_split_patterns(parent, 3.0, CONFIG, rail=Rail.UPPER)[1]
    left, right = blocks_for_pattern(parent, pattern, CONFIG)
    assert left.block_id.endswith("::0")
    assert right.block_id.endswith("::1")
    assert sum(left.start[0:1] + left.end[0:1]) / 2.0 < sum(right.start[0:1] + right.end[0:1]) / 2.0
    assert left.u_start > right.u_start


def test_static_x_geometry_metadata_counts_unique_parents_and_candidate_sources() -> None:
    parents = (
        ParentWeld("upper", (0.0, 8.0), (4.0, 8.0)),
        ParentWeld("lower", (0.0, 2.0), (2.0, 2.0)),
        ParentWeld("vertical", (7.0, 8.0), (7.0, 10.0)),
    )
    metadata = x_split_geometry_metadata(parents, CONFIG)
    assert metadata["x_splittable_parent_count"] == 2
    assert metadata["x_split_pattern_count"] >= 4
    assert 0.0 < metadata["x_splittable_process_share"] < 1.0
    assert metadata["max_x_span"] == pytest.approx(4.0)
    assert metadata["mean_x_span"] == pytest.approx(3.0)
    assert sum(metadata["x_split_source_counts"].values()) == metadata["x_split_pattern_count"]


def test_weighted_median_is_leftmost_deterministic_and_clipped() -> None:
    parents = (
        ParentWeld("a", (-2.0, 8.0), (2.0, 8.0)),  # midpoint 0 -> clipped 0.2
        ParentWeld("b", (9.0, 8.0), (11.0, 8.0)),
        ParentWeld("c", (9.0, 4.0), (11.0, 4.0)),
    )
    up, low = frozen_handover_centers(parents, CONFIG)
    assert up == 0.5  # leftmost weighted median clipped to the new window
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


@pytest.mark.parametrize('axis', ('Y','X'))
def test_range_05_retains_all_legacy_cut_geometry_and_ids(axis):
    old=ScientificConfig(delta_x=.2,delta_y=.2)
    parent=ParentWeld('p',(1.,5.),(1.,7.)) if axis=='Y' else ParentWeld('p',(1.,8.),(5.,8.))
    generate=(lambda cfg:generate_y_split_patterns(parent,cfg)) if axis=='Y' else (lambda cfg:generate_x_split_patterns(parent,3.,cfg,rail=Rail.UPPER))
    earlier=generate(old);current=generate(CONFIG)
    for p in earlier:
        assert any(p.point_id==q.point_id and p.t==pytest.approx(q.t) for q in current)
    assert len(current)==5
    assert len({p.point_id for p in current})==len(current)
    assert len({round(p.t,10) for p in current})==len(current)


def test_y_whole_eligibility_and_mandatory_change_with_task_coverage():
    from mrta_reference.geometry import build_legal_pattern_catalog,robot_is_eligible
    from mrta_reference.scope import FORMAL_SCOPE_V2
    old=ScientificConfig(delta_x=.2,delta_y=.2)
    overlap=ParentWeld('overlap',(1.,5.6),(2.,6.4))
    crossing=ParentWeld('crossing',(4.,5.4),(5.,6.6))
    assert not whole_eligible_rails(overlap.start,overlap.end,old)
    assert whole_eligible_rails(overlap.start,overlap.end,CONFIG)==frozenset((Rail.UPPER,Rail.LOWER))
    assert all(p.mandatory for p in generate_y_split_patterns(overlap,old))
    assert not any(p.mandatory for p in generate_y_split_patterns(overlap,CONFIG))
    catalog=build_legal_pattern_catalog((overlap,crossing),CONFIG,FORMAL_SCOPE_V2)
    assert any(p.kind is SplitKind.WHOLE for p in catalog['overlap'])
    assert all(p.kind is SplitKind.Y_SPLIT and p.mandatory for p in catalog['crossing'])
    for parent in (overlap,crossing):
        for pattern in generate_y_split_patterns(parent,CONFIG):
            lower,upper=blocks_for_pattern(parent,pattern,CONFIG)
            assert robot_is_eligible(lower,2,CONFIG) and robot_is_eligible(upper,0,CONFIG)


@pytest.mark.parametrize('length,allowed', ((.398,False),(.4,True)))
def test_y_exact_euclidean_child_minimum(length,allowed):
    parent=ParentWeld('p',(1.,6.-length/2),(1.,6.+length/2))
    patterns=generate_y_split_patterns(parent,CONFIG)
    assert bool(patterns)==allowed
    if allowed:
        assert len(patterns)==1
        assert [b.length for b in blocks_for_pattern(parent,patterns[0],CONFIG)]==pytest.approx([.2,.2])


def test_oblique_cut_uses_euclidean_length_not_axis_projection():
    y=ParentWeld('y',(1.,5.45),(2.,5.55))
    cut=next(p for p in generate_y_split_patterns(y,CONFIG) if p.point_id=='BY_OUTER_LOWER')
    assert min(b.length for b in blocks_for_pattern(y,cut,CONFIG))>.5
    x=ParentWeld('x',(2.,7.),(2.1,8.))
    patterns=generate_x_split_patterns(x,2.05,CONFIG,rail=Rail.UPPER)
    assert patterns and min(b.length for b in blocks_for_pattern(x,patterns[0],CONFIG))>.5


def test_x_endpoints_vertical_and_short_children_filtered_at_05():
    assert not generate_x_split_patterns(ParentWeld('v',(3.,7.),(3.,9.)),3.,CONFIG,rail=Rail.UPPER)
    parent=ParentWeld('p',(2.5,8.),(3.5,8.))
    points=[parent.point(p.t)[0] for p in generate_x_split_patterns(parent,3.,CONFIG,rail=Rail.UPPER)]
    assert points==pytest.approx([2.8,3.,3.2])
    for length,allowed in ((.398,False),(.4,True)):
        short=ParentWeld('short',(3.-length/2,8.),(3.+length/2,8.))
        cuts=generate_x_split_patterns(short,3.,CONFIG,rail=Rail.UPPER)
        assert bool(cuts)==allowed


def test_y_expansion_changes_rail_specific_weighted_centers():
    parents=(ParentWeld('u',(2.,8.),(4.,8.)),ParentWeld('l',(14.,2.),(16.,2.)),
             ParentWeld('shared',(7.,5.6),(13.,6.4)))
    old=ScientificConfig(delta_x=.2,delta_y=.2)
    assert frozen_handover_centers(parents,old)==(3.,15.)
    assert frozen_handover_centers(parents,CONFIG)==(10.,10.)
    assert frozen_handover_centers(tuple(reversed(parents)),CONFIG)==(10.,10.)


@pytest.mark.parametrize('rail,robots,y', ((Rail.UPPER,(0,1),8.),(Rail.LOWER,(2,3),2.)))
def test_outer_x_canonical_catalog_and_fixed_robot_pair(rail,robots,y):
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.geometry import build_legal_pattern_catalog
    parent=ParentWeld('p',(1.,y),(5.,y))
    catalog=build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)
    pattern=next(p for p in catalog['p'] if p.point_id=='BX_OUTER_LOWER')
    solution=canonicalize((parent,),(pattern,),{robots[0]:('p::0',),robots[1]:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    assert solution.patterns==(pattern,)
    with pytest.raises(ValueError,match='spatial children'):
        canonicalize((parent,),(pattern,),{robots[1]:('p::0',),robots[0]:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    with pytest.raises(ValueError,match='absent from legal catalog'):
        canonicalize((parent,),(pattern,),{robots[0]:('p::0',),robots[1]:('p::1',)},ScientificConfig(delta_x=.2,delta_y=.2),scope=FORMAL_SCOPE_V2)


@pytest.mark.parametrize('point_id', ('BY_OUTER_LOWER','BY_LOWER','BY_CENTER','BY_UPPER','BY_OUTER_UPPER'))
def test_new_y_cut_is_canonical_and_independently_certified(point_id):
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import ScheduleStatus
    parent=ParentWeld('p',(2.,4.),(2.,8.))
    pattern=next(p for p in generate_y_split_patterns(parent,CONFIG) if p.point_id==point_id)
    sol=canonicalize((parent,),(pattern,),{2:('p::0',),0:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    schedule=FormalReferenceEvaluator(FORMAL_SCOPE_V2)(sol,CONFIG,orientations={0:(1,),2:(0,)})
    assert schedule.status is ScheduleStatus.FEASIBLE
    assert certify_schedule(sol,schedule,CONFIG,scope=FORMAL_SCOPE_V2).certified
    forged=replace(schedule,operations=())
    assert not certify_schedule(sol,forged,CONFIG,scope=FORMAL_SCOPE_V2).certified


def test_outer_x_shared_point_has_no_continuous_interference_exception():
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import OperationKind,ScheduleStatus
    parent=ParentWeld('p',(1.,8.),(5.,8.))
    pattern=next(p for p in generate_x_split_patterns(parent,3.,CONFIG,rail=Rail.UPPER) if p.point_id=='BX_OUTER_LOWER')
    sol=canonicalize((parent,),(pattern,),{0:('p::0',),1:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    schedule=FormalReferenceEvaluator(FORMAL_SCOPE_V2)(sol,CONFIG,orientations={0:(0,),1:(1,)})
    assert schedule.status is ScheduleStatus.FEASIBLE
    assert certify_schedule(sol,schedule,CONFIG,scope=FORMAL_SCOPE_V2).certified
    # Forge both children reaching the common cut simultaneously.
    offsets={}
    for robot in (0,1):
        offsets[robot]=min(o.start_time for o in schedule.operations if o.robot_id==robot and o.kind is OperationKind.SETUP)
    left,right=blocks_for_pattern(parent,pattern,CONFIG)
    delay={0:(right.length-left.length)/CONFIG.weld_speed,1:0.}
    operations=tuple(replace(o,start_time=o.start_time-offsets[o.robot_id]+delay[o.robot_id],end_time=o.end_time-offsets[o.robot_id]+delay[o.robot_id])
                     for o in schedule.operations if o.kind is not OperationKind.WAIT)
    forged=replace(schedule,operations=operations)
    report=certify_schedule(sol,forged,CONFIG,scope=FORMAL_SCOPE_V2)
    assert not report.certified
    assert any('interference:' in error or 'order violation' in error for error in report.errors)


def test_three_solvers_really_consume_same_expanded_catalog():
    import mrta_reference.geometry as geometry
    import mrta_search.neighborhood as alns
    import mrta_baselines.hga as hga
    import mrta_baselines.wag_vns as wag
    from mrta_reference.scope import FORMAL_SCOPE_V2
    assert alns.build_legal_pattern_catalog is hga.build_legal_pattern_catalog is wag.build_legal_pattern_catalog is geometry.build_legal_pattern_catalog
    parent=ParentWeld('p',(1.,8.),(5.,8.))
    catalog=geometry.build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)
    assert {'BX_OUTER_LOWER','BX_OUTER_UPPER'}<={p.point_id for p in catalog['p']}
    # Existing three-method access test executes all three production consumers.
    from test_hga_baseline import test_v2_three_method_pattern_access_uses_one_catalog_hash
    test_v2_three_method_pattern_access_uses_one_catalog_hash()



def test_new_outer_x_does_not_replace_existing_midpoint_identity():
    parent=ParentWeld('p',(1.,8.),(4.,8.))
    old=generate_x_split_patterns(parent,3.,ScientificConfig(delta_x=.2,delta_y=.2),rail=Rail.UPPER)
    new=generate_x_split_patterns(parent,3.,CONFIG,rail=Rail.UPPER)
    assert next(p for p in old if p.t==.5).point_id=='MIDPOINT'
    assert next(p for p in new if p.t==.5).point_id=='MIDPOINT'
