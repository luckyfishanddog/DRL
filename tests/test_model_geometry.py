from __future__ import annotations


def test_fixed_euclidean_strict_lengths_and_conservation():
    import math
    import pytest
    from mrta_reference.fixed_taskset import generate_taskset, long_segment_count
    from mrta_reference.model import ParentWeld
    for length, n in ((4.8,1),(5.,1),(5.1,5),(6.,5),(6.6,6),(10.2,10)):
        assert long_segment_count(length) == n
        ts = generate_taskset((ParentWeld('p',(0.,9.),(length,9.)),))
        assert len(ts.tasks) == n
        assert sum(t.length for t in ts.tasks) == pytest.approx(length)
        if n > 1:
            assert all(t.length > 1. for t in ts.tasks)
    ts = generate_taskset((ParentWeld('diag',(0.,6.6),(4.,10.6)),))
    assert len(ts.tasks) == 5  # Euclidean >5, X projection only 4.
    assert sum(t.length for t in ts.tasks) == pytest.approx(math.sqrt(32.))
    upper = generate_taskset((ParentWeld('upper',(0.,5.5),(12.,7.5)),))
    assert any(max(t.start[1],t.end[1]) < 6.5 for t in upper.tasks)
    assert all(t.eligible_robots == (0,1) for t in upper.tasks)


def test_fixed_y_maximin_double_threshold_and_both_rails():
    import pytest
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_reference.model import ParentWeld, ScientificConfig
    ts = generate_taskset((ParentWeld('preferred',(2.,3.),(2.,8.)),
                           ParentWeld('relaxed',(3.,5.4),(3.,6.6)),
                           ParentWeld('both',(1.,5.6),(3.,6.4)),
                           ParentWeld('reachable',(6.,6.),(6.,9.))))
    assert ts.statistics['y_split_count'] == 2
    assert ts.statistics['y_strict_1m_count'] == 1
    assert ts.statistics['y_relaxed_0_5m_count'] == 1
    for p in ('preferred','relaxed'):
        children = [t for t in ts.tasks if t.source_parent_id == p]
        assert len(children) == 2
        assert children[0].u_end == pytest.approx(.5 if p == 'relaxed' else .5)
        assert 5.5 <= children[0].end[1] <= 6.5
        assert min(t.length for t in children) > (.5 if p == 'relaxed' else 1.)
    assert next(t for t in ts.tasks if t.source_parent_id == 'both').eligible_robots == (0,1,2,3)
    assert len([t for t in ts.tasks if t.source_parent_id == 'reachable']) == 1
    with pytest.raises(ValueError, match='impossible.*start='):
        generate_taskset((ParentWeld('impossible',(1.,5.98),(1.,6.02)),),ScientificConfig(delta_y=.01))


def test_fixed_merges_owned_runs_before_ordering_and_coordination_counts_setup_once():
    import pytest
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_reference.model import ParentWeld, ScientificConfig, OperationKind
    from mrta_reference.solution import canonicalize, block_map
    from mrta_reference.scheduler import native_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    c = ScientificConfig()
    ts = generate_taskset((ParentWeld('p',(0.,9.),(6.6,9.)),ParentWeld('q',(9.,9.),(10.,9.))))
    ids = [t.task_id for t in ts.tasks if t.source_parent_id == 'p']
    # Deliberately separate all sibling atoms in the provisional ordering.
    s = canonicalize(ts.parents,ts.patterns,{0:(ids[0],'q::seg_000',*ids[1:])},c,scope=FORMAL_SCOPE_V2)
    assert s.routes[0].block_ids == ('p::job_000_006','q::seg_000')
    blocks = block_map(s,c,scope=FORMAL_SCOPE_V2)
    assert blocks['p::job_000_006'].length == pytest.approx(6.6)
    result = native_schedule(s,c,scope=FORMAL_SCOPE_V2,orientations={0:(0,0),1:(),2:(),3:()})
    assert result.feasible
    assert sum(o.kind is OperationKind.SETUP for o in result.operations) == 2
    assert sum(o.kind is OperationKind.POST for o in result.operations) == 2
    assert result.cmax == pytest.approx(c.process_time(6.6)+c.process_time(1.)+2.4/c.empty_speed)
    s = canonicalize(ts.parents,ts.patterns,{0:(ids[0],*ids[2:],'q::seg_000'),1:(ids[1],)},c)
    assert s.routes[0].block_ids[:2] == (ids[0],'p::job_002_006')
    with pytest.raises(ValueError,match='more than once'):
        canonicalize(ts.parents,ts.patterns,{0:(*ids,'q::seg_000',ids[0])},c)
    with pytest.raises(ValueError,match='missing fixed atom'):
        canonicalize(ts.parents,ts.patterns,{0:(*ids[:-1],'q::seg_000')},c)
    with pytest.raises(ValueError,match='ineligible'):
        canonicalize(ts.parents,ts.patterns,{2:(*ids,'q::seg_000')},c)


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
    assert scope.reference_scheduler_policy_id == 'NATIVE_PRIORITY_FOUR_ROBOTS'
    assert len({FORMAL_SCOPE_V1.scope_hash,FORMAL_SCOPE_V1_1.scope_hash,FORMAL_SCOPE_V2.scope_hash}) == 3
    assert scope.scope_hash == FormalScope().scope_hash
    assert scope.scope_hash == hashlib.sha256(scope.canonical_json.encode()).hexdigest()
    for change in (
        {"optional_x_split_policy": "INCLUDED"}, {"pattern_domain": ("WHOLE",)},
        {"terminal_policy": "OCCUPY_TO_CMAX"}, {"empty_route_policy": "PARK"},
        {"initial_deployment_policy": "PAID"},
        {"reference_scheduler_policy_id": "OTHER"},
    ):
        mutated = replace(scope, **change)
        assert mutated.scope_hash != scope.scope_hash
        with pytest.raises(ValueError, match="unsupported"):
            mutated.validate_implemented()
    with pytest.raises(FrozenInstanceError):
        scope.terminal_policy = 'OTHER'
    assert ACTIVE_FORMAL_SCOPE is FORMAL_SCOPE_V2
    assert FORMAL_SCOPE_V1_1.scope_hash != FORMAL_SCOPE_V1.scope_hash
    assert not any('budget' in name or 'recovery' in name for name in scope.__dataclass_fields__)
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
        "BX_OUTER_LOWER", "BX_CENTER", "BX_OUTER_UPPER", "MIDPOINT"
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
    assert CONFIG.min_child_length == 0.5
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
    assert [candidate.point_id for candidate in candidates] == ["BY_OUTER_LOWER", "BY_CENTER", "BY_OUTER_UPPER"]
    assert [candidate.t for candidate in candidates] == pytest.approx([0.25, 0.5, 0.75])
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
    pattern = SplitPattern("p", SplitKind.X_SPLIT, 0.5, "given-x", rail=Rail.UPPER)
    blocks = blocks_for_pattern(parent, pattern, CONFIG)
    assert len(blocks) == 2
    assert sum(block.length for block in blocks) == pytest.approx(parent.length)
    assert min(block.length for block in blocks) >= CONFIG.min_child_length
    with pytest.raises(ValueError, match="Lmin"):
        blocks_for_pattern(parent, SplitPattern("p", SplitKind.X_SPLIT, 0.499, "short", rail=Rail.UPPER), CONFIG)

    exactly = blocks_for_pattern(
        parent, SplitPattern("p", SplitKind.X_SPLIT, 0.5, "exact-lmin", rail=Rail.UPPER), CONFIG
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
    assert [item.point_id for item in first] == ["BX_OUTER_LOWER", "BX_CENTER", "BX_OUTER_UPPER"]
    assert [item.t for item in first] == pytest.approx([0.375, 0.5, 0.625])
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
def test_range_05_removes_legacy_inner_cuts_but_keeps_center(axis):
    old=ScientificConfig(delta_x=.2,delta_y=.2)
    parent=ParentWeld('p',(1.,5.),(1.,7.)) if axis=='Y' else ParentWeld('p',(1.,8.),(5.,8.))
    generate=(lambda cfg:generate_y_split_patterns(parent,cfg)) if axis=='Y' else (lambda cfg:generate_x_split_patterns(parent,3.,cfg,rail=Rail.UPPER))
    earlier=generate(old);current=generate(CONFIG)
    center=next(p for p in earlier if p.point_id.endswith('_CENTER'))
    assert any(center.point_id==q.point_id and center.t==pytest.approx(q.t) for q in current)
    for p in earlier:
        if p.point_id.endswith(('_LOWER','_UPPER')):
            assert not any(p.t==pytest.approx(q.t) for q in current)
    assert len(current)==3
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


@pytest.mark.parametrize('length,allowed', ((.998,False),(1.,True)))
def test_y_exact_euclidean_child_minimum(length,allowed):
    parent=ParentWeld('p',(1.,6.-length/2),(1.,6.+length/2))
    patterns=generate_y_split_patterns(parent,CONFIG)
    assert bool(patterns)==allowed
    if allowed:
        assert len(patterns)==1
        assert [b.length for b in blocks_for_pattern(parent,patterns[0],CONFIG)]==pytest.approx([.5,.5])


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
    assert points==pytest.approx([3.])
    for length,allowed in ((.8,False),(.998,False),(1.,True)):
        short=ParentWeld('short',(3.-length/2,8.),(3.+length/2,8.))
        cuts=generate_x_split_patterns(short,3.,CONFIG,rail=Rail.UPPER)
        assert bool(cuts)==allowed


def test_short_whole_and_mandatory_y_without_legal_cut():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    short = ParentWeld('short', (1., 8.), (1.8, 8.))
    catalog = build_legal_pattern_catalog((short,), CONFIG, FORMAL_SCOPE_V2)
    assert [p.kind for p in catalog['short']] == [SplitKind.WHOLE]
    # Use a smaller overlap only to construct a mandatory infeasibility fixture.
    config = ScientificConfig(delta_y=.2)
    crossing = ParentWeld('crossing', (1., 5.7), (1., 6.3))
    with pytest.raises(ValueError, match='mandatory Y family is empty'):
        build_legal_pattern_catalog((crossing,), config, FORMAL_SCOPE_V2)


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


@pytest.mark.parametrize('point_id', ('BY_OUTER_LOWER','BY_CENTER','BY_OUTER_UPPER'))
def test_new_y_cut_is_canonical_and_independently_certified(point_id):
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import NativeReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import ScheduleStatus
    parent=ParentWeld('p',(2.,4.),(2.,8.))
    pattern=next(p for p in generate_y_split_patterns(parent,CONFIG) if p.point_id==point_id)
    sol=canonicalize((parent,),(pattern,),{2:('p::0',),0:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    schedule=NativeReferenceEvaluator(FORMAL_SCOPE_V2)(sol,CONFIG,orientations={0:(1,),2:(0,)})
    assert schedule.status is ScheduleStatus.FEASIBLE
    assert certify_schedule(sol,schedule,CONFIG,scope=FORMAL_SCOPE_V2).certified
    forged=replace(schedule,operations=())
    assert not certify_schedule(sol,forged,CONFIG,scope=FORMAL_SCOPE_V2).certified


@pytest.mark.parametrize('mid_y', (5.8,6.2))
def test_midpoint_on_removed_y_line_keeps_its_new_identity_and_certifies(mid_y):
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import NativeReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    parent=ParentWeld('p',(1.,mid_y-1.),(5.,mid_y+1.))
    pattern=next(p for p in generate_y_split_patterns(parent,CONFIG) if p.point_id=='MIDPOINT')
    sol=canonicalize((parent,),(pattern,),{2:('p::0',),0:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    schedule=NativeReferenceEvaluator(FORMAL_SCOPE_V2)(sol,CONFIG,orientations={0:(1,),2:(0,)})
    report=certify_schedule(sol,schedule,CONFIG,scope=FORMAL_SCOPE_V2)
    assert report.certified,report.errors


def test_outer_x_shared_point_has_no_continuous_interference_exception():
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.scheduler import NativeReferenceEvaluator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import OperationKind,ScheduleStatus
    parent=ParentWeld('p',(1.,8.),(5.,8.))
    pattern=next(p for p in generate_x_split_patterns(parent,3.,CONFIG,rail=Rail.UPPER) if p.point_id=='BX_OUTER_LOWER')
    sol=canonicalize((parent,),(pattern,),{0:('p::0',),1:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    schedule=NativeReferenceEvaluator(FORMAL_SCOPE_V2)(sol,CONFIG,orientations={0:(0,),1:(1,)})
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


@pytest.mark.parametrize('axis', ('Y', 'X'))
def test_removed_inner_cut_cannot_reenter_formal_catalog(axis):
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    if axis=='Y':
        parent=ParentWeld('p',(2.,5.),(2.,7.))
        pattern=SplitPattern('p',SplitKind.Y_SPLIT,.4,'BY_LOWER',True)
        routes={2:('p::0',),0:('p::1',)}
    else:
        parent=ParentWeld('p',(1.,8.),(5.,8.))
        pattern=SplitPattern('p',SplitKind.X_SPLIT,.45,'BX_LOWER',rail=Rail.UPPER)
        routes={0:('p::0',),1:('p::1',)}
    with pytest.raises(ValueError,match='absent from legal catalog'):
        canonicalize((parent,),(pattern,),routes,CONFIG,scope=FORMAL_SCOPE_V2)


def test_fixed_route_neighborhood_no_longer_builds_dynamic_split_catalog(monkeypatch):
    import mrta_reference.geometry as geometry
    from mrta_reference.model import MoveType
    from mrta_search.neighborhood import applicable_move_mask, make_raw_candidate
    from mrta_reference.scope import FORMAL_SCOPE_V2
    def forbidden(*args, **kwargs):
        raise AssertionError('fixed route candidates must not generate dynamic split alternatives')
    monkeypatch.setattr(geometry, 'generate_y_split_patterns', forbidden)
    monkeypatch.setattr(geometry, 'generate_x_split_patterns', forbidden)
    w, state = _br_fixed_fixture()
    solution = w.solution(state)
    mask = applicable_move_mask(solution, w.config, scope=FORMAL_SCOPE_V2)
    retired = (MoveType.SPLIT_ACTIVATE,MoveType.SPLIT_DEACTIVATE,MoveType.SPLIT_POINT_SWITCH)
    assert not set(retired).intersection(mask)
    for move in retired:
        assert make_raw_candidate(solution,w.config,move,seed=7,ordinal=0).candidate is None



def test_new_outer_x_does_not_replace_existing_midpoint_identity():
    parent=ParentWeld('p',(1.,8.),(4.,8.))
    old=generate_x_split_patterns(parent,3.,ScientificConfig(delta_x=.2,delta_y=.2),rail=Rail.UPPER)
    new=generate_x_split_patterns(parent,3.,CONFIG,rail=Rail.UPPER)
    assert next(p for p in old if p.t==.5).point_id=='MIDPOINT'
    assert next(p for p in new if p.t==.5).point_id=='MIDPOINT'


def _br_fixed_fixture():
    import numpy as np
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_reference.model import ParentWeld, ScientificConfig
    from mrta_search.gnn_vns_br import FixedWorkspace
    c = ScientificConfig()
    ts = generate_taskset((
        ParentWeld('p', (0., 9.), (6.6, 9.)),
        ParentWeld('a', (2., 10.), (3., 10.)),
        ParentWeld('b', (7., 9.), (8., 9.)),
        ParentWeld('c', (12., 9.), (13., 9.)),
        ParentWeld('d', (16., 9.), (17., 9.)),
        ParentWeld('l', (1., 2.), (2., 2.)),
        ParentWeld('r', (15., 2.), (16., 2.))))
    w = FixedWorkspace(ts, c)
    names = [('p::job_000_003', 'a::seg_000', 'b::seg_000'),
             ('p::job_003_006', 'c::seg_000', 'd::seg_000'),
             ('l::seg_000',), ('r::seg_000',)]
    routes = [np.array([w.index[n] for n in row], np.int64) for row in names]
    bits = [np.array([i % 2 for i in range(len(row))], np.int64) for row in names]
    return w, w.evaluate(routes, bits)


def test_br_six_actions_match_independent_fixed_canonical_rebuild_and_native_score():
    import numpy as np
    import pytest
    from mrta_baselines.wag_vns import priority_schedule
    from mrta_reference.solution import canonicalize
    w, state = _br_fixed_fixture()
    original_patterns = w.taskset.patterns
    for move in range(1, 7):
        ru, iu, rv, iv = (0, 0, 0, 2) if move == 4 else (0, 0, 1, 1)
        action = w.apply(state, move, ru, iu, rv, iv)
        assert action is not None
        raw = [list(map(int, r)) for r in state.routes]
        u, v = raw[ru][iu], raw[rv][iv]
        if move == 1:
            raw[ru].pop(iu); raw[rv].insert(iv + 1, u)
        elif move == 2:
            raw[ru][iu], raw[rv][iv] = v, u
        elif move == 3:
            a, b = raw[ru][iu:iu + 2], raw[rv][iv:iv + 2]
            raw[ru][iu:iu + 2], raw[rv][iv:iv + 2] = b, a
        elif move == 4:
            raw[ru][iu + 1:iv + 1] = reversed(raw[ru][iu + 1:iv + 1])
            assert action.bits[ru][1:].tolist() == (1 - state.bits[ru][1:][::-1]).tolist()
        elif move == 5:
            a, b = raw[ru][:], raw[rv][:]
            raw[ru] = a[:iu + 1] + b[:iv + 1][::-1]
            raw[rv] = a[iu + 1:][::-1] + b[iv + 1:]
        else:
            a, b = raw[ru][iu + 1:], raw[rv][iv + 1:]
            raw[ru][iu + 1:], raw[rv][iv + 1:] = b, a
        expected = canonicalize(w.taskset.parents, original_patterns,
            {r: tuple(w.numeric.ids[n] for n in row) for r, row in enumerate(raw)}, w.config)
        numeric = w.evaluate(action.routes, action.bits)
        w.validate(action.routes, action.bits)
        actual = w.solution(numeric)
        assert actual == expected
        assert actual.patterns == original_patterns
        public = priority_schedule(actual, w.config, directions=action.bits)
        assert public.feasible == numeric.feasible
        if public.feasible:
            assert numeric.value == pytest.approx(public.cmax)
            assert np.allclose(numeric.completion, public.robot_completion)
    # An action can create a job on an empty route, while a cross-rail illegal
    # transfer is rejected before scheduling rather than relaxed for speed.
    assert w.apply(state, 1, 0, 0, 2, 0) is None


def test_br_shared_gat_pairwise_gradients_and_infeasible_label_order():
    import random
    import torch
    from mrta_search.gnn_vns_br import (
        BRConfig, BottleneckGAT, candidate_actions, local_graph, ranking_pairs,
    )
    torch.set_num_threads(1)
    w, state = _br_fixed_fixture()
    rng = random.Random(17)
    actions = candidate_actions(w, state, BRConfig(candidates=6), rng)
    graph = local_graph(w, state, actions)
    assert graph.nodes.shape == (len(w.taskset.tasks) + 4, 12)
    assert graph.edge_attr.shape[1] == 4
    assert graph.action_features.shape[1] == 16
    pairs = ranking_pairs([3., float('inf'), 2., 3.], rng)
    assert len(pairs) == 5
    assert all(b != 2 for a, b in pairs.tolist())
    assert not len(ranking_pairs([float('inf'),float('inf')],rng))
    numeric = ranking_pairs([3.,float('nan'),2.,float('inf')],rng)
    assert len(numeric)==3 and all(1 not in pair for pair in numeric.tolist())
    model = BottleneckGAT()
    scores = model(graph)
    loss = torch.nn.functional.softplus(scores[1:] - scores[0]).mean()
    loss.backward()
    for layer in (model.layer1, model.layer2):
        assert layer.linear.weight.grad.abs().sum() > 0
        assert layer.edge_bias.weight.grad.abs().sum() > 0
    for projection in (model.task_projection, model.robot_projection):
        assert projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(scores).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.scorer[-1].out_features == 1
    assert {a.move for a in actions} == set(range(1, 7))


def test_br_search_is_independent_of_baseline_solvers_and_full_vnd(monkeypatch):
    import mrta_baselines.hga as hga
    import mrta_baselines.wag_vns as wag
    from mrta_search.gnn_vns_br import BRConfig, BottleneckGAT, run_gnn_vns_br
    def forbidden(*args, **kwargs):
        raise AssertionError('new method must not call a baseline solver or full VND')
    monkeypatch.setattr(hga, 'run_adapted_hga', forbidden)
    monkeypatch.setattr(hga, '_vnd_oriented_arrays', forbidden)
    monkeypatch.setattr(wag, 'run_adapted_wag_vns', forbidden)
    w, state = _br_fixed_fixture()
    workspace, result, record = run_gnn_vns_br(w.taskset, BottleneckGAT(), seed=18,
        settings=BRConfig(candidates=4, top_k=2, max_rounds=2, perturbations=0,
                          initial_attempts=2, time_limit=30.))
    assert workspace.validate(result.routes, result.bits)
    assert record['native_evaluations'] >= 2
    assert record['graph_encodings'] <= 2
    assert workspace.solution(result).patterns == w.taskset.patterns


@pytest.mark.parametrize('dataset,variant', [('original','full_gnn'),
    ('scale1000','full_gnn'),('scale1000','no_edge_gnn'),('scale1000','action_mlp')])
def test_br_final_result_reuses_native_arrays_without_another_scheduler(monkeypatch,dataset,variant):
    import importlib
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module('run_gnn_vns_br')
    w, state = _br_fixed_fixture()
    from mrta_search import gnn_vns_br as br
    from mrta_baselines import wag_vns
    from mrta_reference import scheduler
    def forbidden(*a, **k):
        raise AssertionError('final result must package its saved native arrays')
    monkeypatch.setattr(br, '_priority_arrays', forbidden)
    monkeypatch.setattr(wag_vns, '_priority_arrays', forbidden)
    monkeypatch.setattr(scheduler, 'reference_schedule_optimized', forbidden)
    evaluations = w.evaluations
    monkeypatch.setattr(runner, 'run_gnn_vns_br', lambda *a, **k:
        (w, state, dict(rounds=0, termination_reason='SMOKE')))
    checkpoint = dict(settings=runner.asdict(runner.BRConfig()),
                      node_features=runner.NODE_FEATURES,edge_features=runner.EDGE_FEATURES,
                      action_features=runner.ACTION_FEATURES,
                      robot_features=runner.ROBOT_FEATURES,edge_relations=runner.EDGE_RELATIONS,
                      normalization=runner.asdict(runner.FeatureConfig()),
                      training=dict(learning=dict(selected_epoch=1), total_offline_s=1.))
    checkpoint['learning'] = {variant:dict(selected_epoch=1)}
    row = runner.run_one('train', 'toy', w.taskset, 17, model=object(), checkpoint=checkpoint,
                         dataset=dataset,variant=variant)
    import json
    detail = json.loads(row['diagnostic'])
    assert row['coordination_calls'] == w.evaluations == evaluations
    assert detail['final_scheduler_extra_calls'] == 0 and detail['final_schedule_reused']
    assert row['status'] == state.status.value
    assert row['cmax_s'] == (state.value if state.feasible else None)
    saved = json.loads(row['solution'])
    assert all(type(bit) is int for bits in saved['directions'] for bit in bits)


def test_br_m1_can_reassign_existing_end_atoms_inside_merged_jobs():
    from mrta_reference.fixed_taskset import expand_routes
    w, state = _br_fixed_fixture()
    old = w.solution(state)
    before = {a: r for r, atoms in enumerate(expand_routes(old.routes)) for a in atoms}
    for portion, moved_atom in ((1, 'p::seg_000'), (-1, 'p::seg_002')):
        action = w.apply(state, 1, 0, 0, 1, 1, portion=portion)
        assert action is not None
        w.validate(action.routes, action.bits)
        candidate = w.solution(w.evaluate(action.routes, action.bits))
        after = {a: r for r, atoms in enumerate(expand_routes(candidate.routes)) for a in atoms}
        assert set(after) == set(before)
        assert [a for a in before if before[a] != after[a]] == [moved_atom]
        assert after[moved_atom] == 1
        assert candidate.patterns == old.patterns == w.taskset.patterns


def test_br_train_smoke_entry_leaves_validation_and_test_untouched(monkeypatch):
    import importlib
    from pathlib import Path
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    runner = importlib.import_module('run_gnn_vns_br')
    def forbidden():
        raise AssertionError('no model or non-TRAIN instance should be loaded')
    monkeypatch.setattr(runner, 'load_model', forbidden)
    for split in ('validation','test'):
        with pytest.raises(ValueError, match='TRAIN smoke only'):
            runner.run_one(split,'unused',None,17)


def test_br_edge_graph_has_separate_job_robot_features_and_directed_costs():
    import random
    import numpy as np
    import pytest
    from mrta_search.gnn_vns_br import BRConfig, NODE_FEATURES, EDGE_FEATURES, candidate_actions, local_graph
    w, state = _br_fixed_fixture()
    actions = candidate_actions(w,state,BRConfig(),random.Random(13))
    graph = local_graph(w,state,actions)
    assert len(NODE_FEATURES)==12 and len(EDGE_FEATURES)==4
    robot_base = len(graph.nodes)-4
    assert robot_base==len(w.taskset.tasks)
    assert np.all(graph.nodes[:robot_base,0].numpy()==1)
    assert np.array_equal(graph.nodes[-4:,2:6].numpy(),np.eye(4))
    assert np.allclose(graph.nodes[-4:,8].numpy(),state.loads/w.normalization.time_ref)
    node_ids=dict(enumerate(graph.atom_nodes))
    owners={j:r for r in range(4) for n in state.routes[r] for j in w.job_atoms[n]}
    seen_robot_edges = {j:set() for j in range(robot_base)}
    attrs = {}
    asymmetric = False
    for (a,b),kind,attr in zip(graph.edges.t().tolist(),graph.relations.tolist(),graph.edge_attr.tolist()):
        attrs[a,b,kind]=attr
        if kind==1 and a<robot_base:
            n=node_ids[a];r=b-robot_base
            seen_robot_edges[a].add(r)
            assert attr[0]==float(w.numeric.eligible[n,r])
            assert attr[1]==float(owners[a]==r)
            tail,bit=int(state.routes[r][-1]),int(state.bits[r][-1])
            assert attr[2:]==pytest.approx(w.numeric.travel[tail,n,bit,:]/w.normalization.time_ref)
        elif kind==6:
            n,m=node_ids[a],node_ids[b]
            expected=w.numeric.travel[n,m].reshape(4)/w.normalization.time_ref
            assert attr==pytest.approx(expected)
        elif kind in (2,3):
            assert attr==[0.,0.,0.,0.]
            assert graph.parent_ids[a]==graph.parent_ids[b]
            assert abs(a-b)==1
    assert all(robots==set(range(4)) for robots in seen_robot_edges.values())
    assert set(graph.relations.tolist())==set(range(7))
    for r in range(4):
        for i in range(1,len(state.routes[r])):
            a,b=map(int,state.routes[r][i-1:i+1]);da,db=map(int,state.bits[r][i-1:i+1])
            aa=w.job_atoms[a][::1 if da==0 else -1]
            bb=w.job_atoms[b][::1 if db==0 else -1]
            forward=attrs[aa[-1],bb[0],4][2]
            reverse=attrs[bb[-1],aa[0],5][2]
            assert forward==pytest.approx(w.numeric.travel[a,b,da,db]/w.normalization.time_ref)
            assert reverse==pytest.approx(w.numeric.travel[b,a,db,da]/w.normalization.time_ref)
            asymmetric |= abs(forward-reverse)>1.e-7
    assert asymmetric


def test_br_job_robot_edges_use_inherited_eligibility_inside_shared_band():
    import random
    import numpy as np
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_reference.model import ParentWeld,ScientificConfig
    from mrta_search.gnn_vns_br import FixedWorkspace,BRConfig,candidate_actions,local_graph
    c=ScientificConfig()
    ts=generate_taskset((ParentWeld('upper',(0.,5.5),(12.,7.5)),))
    w=FixedWorkspace(ts,c);n=w.index['upper::seg_000']
    assert max(w.numeric.points[n,:,1])<c.by[1]
    assert w.numeric.eligible[n].tolist()==[True,True,False,False]
    raw=[w.numeric.atomic_nodes[:1],w.numeric.atomic_nodes[1:],np.empty(0,np.int64),np.empty(0,np.int64)]
    ds=[np.zeros(len(r),np.int64) for r in raw]
    from mrta_baselines.hga import _merge_fixed_arrays
    raw,ds=_merge_fixed_arrays(raw,ds,w.numeric.fixed_starts,w.numeric.fixed_stops,w.numeric.fixed_lookup)
    state=w.evaluate(raw,ds)
    actions=candidate_actions(w,state,BRConfig(),random.Random(4))
    graph=local_graph(w,state,actions);base=len(graph.nodes)-4
    job_node=graph.atom_nodes.index(n)
    for robot in range(4):
        edge=next(k for k,(a,b) in enumerate(graph.edges.t().tolist()) if a==job_node and b==base+robot)
        assert graph.edge_attr[edge,0].item()==float(robot<2)


def test_br_edge_attributes_change_attention_and_receive_gradients():
    import torch
    from mrta_search.gnn_vns_br import SparseGATLayer
    layer=SparseGATLayer(1,hidden=4,heads=1)
    with torch.no_grad():
        layer.linear.weight.fill_(1.)
        layer.source.zero_();layer.destination.zero_();layer.relation.weight.zero_()
        layer.edge_bias.weight.zero_();layer.edge_bias.weight[0,2]=2.
    x=torch.tensor([[0.],[1.],[2.]])
    edges=torch.tensor([[0,1],[2,2]])
    relation=torch.tensor([5,5])
    plain=torch.zeros((2,4))
    changed=plain.clone();changed[1,2]=1.;changed.requires_grad_()
    a=layer(x,edges,relation,plain)[2]
    b=layer(x,edges,relation,changed)[2]
    assert torch.all(b>a)
    b.sum().backward()
    assert changed.grad[:,2].abs().sum()>0
    with torch.no_grad():
        layer.edge_bias.weight.zero_();layer.edge_bias.weight[0,0]=2.
    changed=plain.clone();changed[1,0]=1.
    assert torch.all(layer(x,edges,relation,changed)[2]>a)


def test_br_joint_portions_map_only_the_transferred_atoms_and_refresh_state():
    import numpy as np
    from mrta_search.gnn_vns_br import local_graph
    w,state=_br_fixed_fixture()
    for portion,task_id in ((1,'p::seg_000'),(-1,'p::seg_002')):
        action=w.apply(state,1,0,0,1,1,flip=True,portion=portion)
        assert action.joint==(1,0,0,1,1,True,portion)
        assert action.features[12]==1 and action.features[13:16].sum()==1
        assert action.features[14 if portion==1 else 15]==1
        graph=local_graph(w,state,[action])
        source=graph.action_members[0,graph.action_members[1]==0].tolist()
        assert source==[graph.task_ids.index(task_id)]
        assert action.u!=action.moved
        candidate=w.evaluate(action.routes,action.bits)
        after=local_graph(w,candidate,[])
        j=source[0]
        old_owner=next(k for k,(a,b) in enumerate(graph.edges.t().tolist())
                       if a==j and b==len(w.taskset.tasks))
        new_owner=next(k for k,(a,b) in enumerate(after.edges.t().tolist())
                       if a==j and b==len(w.taskset.tasks)+1)
        assert graph.edge_attr[old_owner,1]==after.edge_attr[new_owner,1]==1
        target_index=next(i for i,n in enumerate(candidate.routes[1]) if j in w.job_atoms[n])
        # If it joins a neighboring owned interval, mandatory merging gives
        # every atom the resulting job's actual execution direction.
        assert after.nodes[j,11]==candidate.bits[1][target_index]
        assert np.array_equal(graph.nodes[:len(w.taskset.tasks),2:9],after.nodes[:len(w.taskset.tasks),2:9])
        assert not np.array_equal(graph.nodes[:,9:12],after.nodes[:,9:12])


def test_br_complete_graph_retains_unassigned_tasks_without_rescheduling(monkeypatch):
    from dataclasses import replace
    import numpy as np
    import mrta_search.gnn_vns_br as br
    from mrta_reference.model import ScheduleStatus
    w,state=_br_fixed_fixture()
    assert w.evaluate(state.routes,state.bits) is state
    calls=w.evaluations
    def forbidden(*a,**k):raise AssertionError('graph construction must not schedule')
    monkeypatch.setattr(br,'_priority_arrays',forbidden)
    partial=replace(state,routes=[np.empty(0,np.int64) for _ in range(4)],
        bits=[np.empty(0,np.int64) for _ in range(4)],value=float('inf'),
        evaluation_status=ScheduleStatus.DEADLOCK,completion=np.zeros(4),loads=np.zeros(4),
        operations=[np.empty((0,9)) for _ in range(4)])
    graph=br.local_graph(w,partial,[])
    assert len(graph.nodes)==len(w.taskset.tasks)+4 and not graph.feedback_valid.any()
    assert not graph.nodes[:-4,9:12].any()
    assert w.evaluations==calls and w.evaluate(state.routes,state.bits) is state
    for r in range(4):
        from mrta_baselines.wag_vns import _home
        expected=2*np.array(_home(r,w.config))/np.array([20.,12.])-1
        assert np.allclose(graph.nodes[-4+r,6:8],expected)
    bad=[route.copy() for route in state.routes];bad[0][0]=-1
    assert w.evaluate(bad,state.bits).status is ScheduleStatus.INFEASIBLE


def test_br_robot_features_use_recorded_move_wait_and_actual_processing():
    import numpy as np
    from mrta_baselines.hga import _merge_fixed_arrays
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_search.gnn_vns_br import FeatureConfig,FixedWorkspace,local_graph
    c=ScientificConfig(weld_speed=1.,empty_speed=100.,t_pre=1.,t_post=1.)
    ts=generate_taskset((ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(17.,9.),(18.,9.)),
                        ParentWeld('c',(14.,9.),(20.,9.))),c)
    w=FixedWorkspace(ts,c,normalization=FeatureConfig(time_ref=10.))
    routes=[np.array([n for n in w.numeric.atomic_nodes if w.numeric.parent_index[n]<2]),
            np.array([n for n in w.numeric.atomic_nodes if w.numeric.parent_index[n]==2]),
            np.empty(0,np.int64),np.empty(0,np.int64)]
    bits=[np.zeros(len(row),np.int64) for row in routes]
    routes,bits=_merge_fixed_arrays(routes,bits,w.numeric.fixed_starts,w.numeric.fixed_stops,w.numeric.fixed_lookup)
    state=w.evaluate(routes,bits)
    assert state.feasible and sum(op[8]==4 for ops in state.operations for op in ops)>0
    graph=local_graph(w,state,[])
    for r,ops in enumerate(state.operations):
        recorded=[sum(op[6]-op[5] for op in ops if op[8]==kind) for kind in (0,4)]
        expected=[sum(c.process_time(w.lengths[n]) for n in state.routes[r]),recorded[0],state.completion[r],recorded[1]]
        assert np.allclose(graph.nodes[-4+r,8:12],np.array(expected)/10.)
    assert graph.feedback_valid.all()


def test_br_small_mixed_initialization_and_uncertainty_are_bounded(monkeypatch):
    import random,time
    import numpy as np
    from mrta_search.gnn_vns_br import BRConfig,BottleneckGAT,action_distribution,route_diversity
    w,_=_br_fixed_fixture()
    model=BottleneckGAT()
    def forbidden(*a,**k):raise AssertionError('untrained weights cannot guide initialization')
    monkeypatch.setattr(model,'forward',forbidden)
    state=w.initialize(random.Random(27),BRConfig(candidates=4),time.perf_counter()+60.,model)
    assert state is w.initial_candidates[0] and 1<=len(w.initial_candidates)<=6
    assert w.initialization['uncertainty'] is None and not w.initialization['gnn_guidance']
    assert w.initialization['diversity']==route_diversity(w.initial_candidates)
    assert len({tuple((tuple(r),tuple(b)) for r,b in zip(s.routes,s.bits)) for s in w.initial_candidates})==len(w.initial_candidates)
    assert w.evaluations<=7  # Includes the fixture's initial state.
    feasible=[s for s in w.initial_candidates if s.feasible]
    if feasible:assert state.value==min(s.value for s in feasible)
    probability,u=action_distribution([0.,0.,0.]);assert np.allclose(probability,1/3) and u==pytest.approx(1.)
    assert action_distribution([100.,0.,0.])[1]<.01
    assert action_distribution([1.])[1]==0.


def test_br_new_checkpoint_requires_both_node_types_relations_and_normalization(monkeypatch):
    import importlib
    from dataclasses import asdict
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    checkpoint=dict(node_features=runner.NODE_FEATURES,robot_features=runner.ROBOT_FEATURES,
        edge_relations=runner.EDGE_RELATIONS,edge_features=runner.EDGE_FEATURES,
        action_features=runner.ACTION_FEATURES,normalization=asdict(runner.FeatureConfig()))
    runner.check_model_features(checkpoint)
    old22=dict(checkpoint,action_features=('M1','M2','M3','M4','M5','M6',
        'source_load','target_load','source_completion','target_completion','source_processing_delta',
        'target_processing_delta','source_empty_delta','target_empty_delta','route_peak_delta',
        'source_route_delta','target_route_delta','source_job_fraction','flip','portion_none','portion_start','portion_end'))
    with pytest.raises(ValueError,match='incompatible checkpoint'):runner.check_model_features(old22)
    for key in ('node_features','robot_features','edge_relations','edge_features','action_features','normalization'):
        old=dict(checkpoint);old.pop(key)
        with pytest.raises(ValueError,match='incompatible checkpoint'):runner.check_model_features(old)
    norm=runner.FeatureConfig(time_ref=1234.)
    checkpoint['normalization']=asdict(norm)
    checkpoint['model']=runner.make_ranker(normalization=norm).state_dict()
    with pytest.raises(ValueError,match='completed training record'):runner.load_model(checkpoint=checkpoint)
    checkpoint['training']=dict(epochs=1,learning=dict(selected_epoch=1))
    model,_=runner.load_model(checkpoint=checkpoint)
    assert model.normalization==norm and model.trained
    assert model.feedback_embedding.weight.shape==(2,64)


def test_br_deadlock_features_and_four_status_pairs_use_no_fabricated_cmax():
    from dataclasses import replace
    import random
    import numpy as np
    from mrta_reference.model import ScheduleStatus
    from mrta_search.gnn_vns_br import local_graph,ranking_pairs
    w,state=_br_fixed_fixture()
    deadlock=replace(state,value=float('inf'),completion=np.zeros(4),evaluation_status=ScheduleStatus.DEADLOCK)
    action=w.apply(deadlock,1,0,0,1,1)
    assert action is not None
    expected=((action.costs-action.loads)-(state.costs-state.loads))[[0,1]]/w.normalization.time_ref
    assert np.allclose(action.features[8:10],expected)
    graph=local_graph(w,deadlock,[action])
    assert not graph.feedback_valid.any() and not graph.nodes[-4:,10].any()
    labels=[replace(state,value=3.,evaluation_status=ScheduleStatus.FEASIBLE),deadlock,
            replace(deadlock,evaluation_status=ScheduleStatus.INFEASIBLE),
            replace(deadlock,value=float('nan'),evaluation_status=ScheduleStatus.NUMERIC_FAILURE)]
    assert ranking_pairs(labels,random.Random(3)).tolist()==[[0,1],[0,2]]


def test_br_action16_order_values_and_shared_gnn_mlp_description():
    import numpy as np
    from mrta_search.gnn_vns_br import ACTION_FEATURES,BottleneckGAT,ActionMLP,local_graph
    assert ACTION_FEATURES==('M1','M2','M3','M4','M5','M6','source_processing_delta',
        'target_processing_delta','source_empty_delta','target_empty_delta','route_peak_delta',
        'source_job_fraction','flip','portion_none','portion_start','portion_end')
    w,state=_br_fixed_fixture()
    action=w.apply(state,1,0,0,1,1,flip=True,portion=1)
    assert action is not None and action.features.shape==(16,)
    expected=[action.loads[0]-state.loads[0],action.loads[1]-state.loads[1],
        (action.costs[0]-action.loads[0])-(state.costs[0]-state.loads[0]),
        (action.costs[1]-action.loads[1])-(state.costs[1]-state.loads[1]),
        max(action.costs)-max(state.costs)]
    assert np.allclose(action.features[6:11],np.array(expected)/w.normalization.time_ref)
    assert action.features[11]==pytest.approx(w.numeric.processing[action.moved]/max(1.,state.loads[0]))
    graph=local_graph(w,state,[action])
    gnn,mlp=BottleneckGAT(),ActionMLP()
    assert gnn.scorer[0].in_features==4*64+16 and mlp.scorer[0].in_features==16
    assert gnn(graph).shape==mlp(graph).shape==(1,)


def test_br_feedback_valid_masks_partial_times_and_changes_robot_embeddings():
    from dataclasses import replace
    import numpy as np
    import torch
    from mrta_reference.model import ScheduleStatus
    from mrta_search.gnn_vns_br import BottleneckGAT,local_graph
    torch.set_num_threads(1);torch.manual_seed(31)
    w,state=_br_fixed_fixture()
    action=w.apply(state,1,0,0,1,1)
    partial_ops=[np.array([[2.,0.,0.,1.,0.,0.,2.,0.,0.],
                          [3.,1.,0.,1.,0.,2.,5.,0.,4.]]) for _ in range(4)]
    partial=replace(state,operations=partial_ops,completion=np.full(4,5.))
    for status in (ScheduleStatus.DEADLOCK,ScheduleStatus.INFEASIBLE,ScheduleStatus.NUMERIC_FAILURE):
        graph=local_graph(w,replace(partial,evaluation_status=status),[action])
        assert graph.nodes.shape[1]==12 and graph.edge_attr.shape[1]==4
        assert graph.feedback_valid.shape==(4,) and not graph.feedback_valid.any()
        assert not graph.nodes[-4:,9:12].any()
        assert np.allclose(graph.nodes[-4:,8],state.loads/w.normalization.time_ref)
    feasible=local_graph(w,replace(partial,evaluation_status=ScheduleStatus.FEASIBLE),[action])
    assert feasible.feedback_valid.all()
    assert np.allclose(feasible.nodes[-4:,9:12],np.array([2.,5.,3.])/w.normalization.time_ref)
    model=BottleneckGAT()
    changed=replace(graph,feedback_valid=torch.ones(4,dtype=torch.bool))
    assert not torch.allclose(model.encode(graph)[-4:],model.encode(changed)[-4:])
    scores=model(graph);scores.square().mean().backward()
    assert torch.isfinite(scores).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.feedback_embedding.weight.grad.abs().sum()>0


def test_br_selected_train_layouts_validate_before_any_native_label_work(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    a,b,c=object(),object(),object()
    monkeypatch.setattr(runner,'training_tasksets',lambda dataset:iter([('train_A',a),('train_B',b),('train_C',c)]))
    assert list(runner.selected_training_tasksets('scale1000',['train_C','train_A']))==[('train_C',c),('train_A',a)]
    monkeypatch.setattr(runner,'fit_training_normalization',lambda *a:runner.FeatureConfig())
    def forbidden(*a,**k):raise AssertionError('invalid selection reached teachers/scheduling')
    monkeypatch.setattr(runner,'read_baselines',forbidden);monkeypatch.setattr(runner,'FixedWorkspace',forbidden)
    for layouts in (['train_A','train_A'],[],['validation_A'],['test_A'],['absent']):
        with pytest.raises(ValueError):runner.build_supervision(dataset='scale1000',layouts=layouts)


def test_br_teachers_require_native_cmax_complete_routes_and_train_status(monkeypatch):
    import importlib,json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    solution=json.dumps(dict(routes=[['a'],[],[],[]],directions=[[0],[],[],[]]))
    valid=dict(split='train',layout_id='train_A',status='FEASIBLE',termination_reason='ITERATION_LIMIT',
               native_cmax_s='100',cmax_s='999',solution=solution)
    rows=[valid,dict(valid,native_cmax_s='200',cmax_s='1')]
    rows.extend(dict(valid,native_cmax_s=value) for value in ('','nan','inf','bad','0','-1'))
    rows.extend([dict(valid,status='DEADLOCK'),dict(valid,termination_reason='ERROR'),
                 dict(valid,split='validation',solution='do not parse'),dict(valid,split='test'),
                 dict(valid,solution='{}'),dict(valid,solution=json.dumps(dict(routes=[['a']],directions=[[0]])))])
    monkeypatch.setattr(runner,'read_baselines',lambda *a,**k:rows)
    teachers=runner.read_training_teachers(['train_A'])
    assert len(teachers)==2 and all(float(row['native_cmax_s'])==100 for row in teachers.values())


def test_br_train_cli_passes_explicit_layout_selection(monkeypatch):
    import importlib,sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br');observed=[]
    monkeypatch.setattr(sys,'argv',['run_gnn_vns_br.py','train','--dataset','scale1000','--layouts','train_A','train_B'])
    monkeypatch.setattr(runner,'train',lambda args:observed.append(args))
    runner.main()
    assert observed[0].layouts==['train_A','train_B'] and observed[0].variant=='full_gnn'


def _br_shared_robot_fixture():
    import numpy as np
    from mrta_reference.fixed_taskset import generate_taskset
    from mrta_reference.model import ParentWeld,ScientificConfig
    from mrta_search.gnn_vns_br import FixedWorkspace
    positions=(1.,3.,5.,8.,10.,12.,14.,16.,18.)
    ts=generate_taskset(tuple(ParentWeld(f'p{i}',(x,6.),(x+.8,6.)) for i,x in enumerate(positions)))
    w=FixedWorkspace(ts,ScientificConfig())
    ids=w.numeric.atomic_nodes
    routes=[ids[:3],ids[3:5],ids[5:7],ids[7:]]
    return w,w.evaluate(routes,[np.zeros(len(r),np.int64) for r in routes])


def test_br_candidates_cover_each_target_robot_and_empty_routes():
    import random
    import numpy as np
    from mrta_search.gnn_vns_br import BRConfig,candidate_actions
    w,state=_br_shared_robot_fixture();settings=BRConfig()
    actions=candidate_actions(w,state,settings,random.Random(8))
    assert {a.move for a in actions}==set(range(1,7))
    assert len(actions)<=6*settings.candidates
    heavy=state.bottleneck
    counts={r:sum(a.move==1 and a.rv==r for a in actions) for r in range(4) if r!=heavy}
    assert min(counts.values())>0 and max(counts.values())-min(counts.values())<=2
    for action in actions:w.validate(action.routes,action.bits)
    routes=[r.copy() for r in state.routes];bits=[d.copy() for d in state.bits]
    routes[heavy]=np.concatenate((routes[heavy],routes[3]));routes[3]=np.empty(0,np.int64)
    bits[heavy]=np.concatenate((bits[heavy],bits[3]));bits[3]=np.empty(0,np.int64)
    empty_state=w.evaluate(routes,bits)
    actions=candidate_actions(w,empty_state,settings,random.Random(8))
    assert any(a.move==1 and a.rv==3 and a.v==-1 for a in actions)
    for action in actions:w.validate(action.routes,action.bits)


@pytest.mark.parametrize('initial_value,values,rounds,shakes,expected',[
    (10.,[9.,8.,7.,6.,5.,4.],1,0,4.),
    (float('inf'),[9.,8.,7.,6.,5.,4.],1,0,4.),
    (10.,[11.,12.,13.,14.,15.,16.],2,1,10.),
])
def test_br_global_best_improvement_and_shaking_preserve_best(monkeypatch,initial_value,values,rounds,shakes,expected):
    import numpy as np
    import torch
    from types import SimpleNamespace
    import mrta_search.gnn_vns_br as br
    def state(value):
        return br.State([],[],value,np.full(4,value),np.ones(4),np.ones(4),[])
    class Workspace:
        def __init__(self,*args,**kwargs):self.evaluations=0;self.visited=[];self.initialization={}
        def initialize(self,*args):self.evaluations+=1;return state(initial_value)
        def evaluate(self,route,bits):
            self.evaluations+=1;self.visited.append(route)
            return state(values[route-1])
        def validate(self,*args):return True
    class Model:
        def eval(self):return self
        def __call__(self,graph):return torch.tensor([6.,5.,4.,3.,2.,1.])
    actions=[SimpleNamespace(move=m,routes=m,bits=[]) for m in range(1,7)]
    monkeypatch.setattr(br,'FixedWorkspace',Workspace)
    monkeypatch.setattr(br,'candidate_actions',lambda *args:actions)
    monkeypatch.setattr(br,'local_graph',lambda *args:None)
    w,best,record=br.run_gnn_vns_br(None,Model(),seed=4,
        settings=br.BRConfig(max_rounds=rounds,perturbations=shakes,initial_attempts=1))
    assert best.value==expected
    assert w.visited[:6]==list(range(1,7))
    assert record['native_evaluations']<=1+18*rounds
    if shakes:assert record['perturbations']==1


def test_br_supervision_uses_online_budgets_greedy_states_and_train_only(monkeypatch):
    import importlib
    import json
    from pathlib import Path
    from mrta_search.gnn_vns_br import BRConfig,candidate_actions,local_graph
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    w,state=_br_fixed_fixture()
    payload=runner.solution_record(w.solution(state),state.bits)
    teacher=dict(split='train',layout_id='train_A',status='FEASIBLE',termination_reason='ITERATION_LIMIT',
                 native_cmax_s='100',solution=json.dumps(payload))
    forbidden=dict(teacher,split='validation',layout_id='validation_A',solution='must not parse')
    test=dict(teacher,split='test',layout_id='test_A',solution='must not parse')
    monkeypatch.setattr(runner,'iter_tasksets',lambda split,dataset='original':
        iter([('train_A',w.taskset)]) if split=='train' else (_ for _ in ()).throw(AssertionError('non-TRAIN read')))
    monkeypatch.setattr(runner,'read_baselines',lambda method,**kwargs:[teacher,forbidden,test])
    observed=[]
    def generate(workspace,current,settings,rng):
        observed.append((settings.alpha,settings.anchors,settings.candidates))
        return candidate_actions(workspace,current,settings,rng)
    monkeypatch.setattr(runner,'candidate_actions',generate)
    examples,stats=runner.build_supervision(seed=4,states_per_source=1,progress=lambda *a:None)
    assert examples and observed and set(observed)=={(8,12,40)}
    assert stats['layouts']==['train_A'] and stats['greedy_initializations']==1
    assert all(name=='train_A' and graph.nodes.shape[1]==12 and graph.edge_attr.shape[1]==4
               for name,graph,pairs in examples)
    assert stats['candidate_parameters']==runner.asdict(BRConfig())


def test_br_three_rankers_mask_edges_and_mlp_ignores_the_graph():
    import random
    from dataclasses import replace
    import torch
    from mrta_search.gnn_vns_br import BRConfig,candidate_actions,local_graph,make_ranker
    torch.set_num_threads(1);torch.manual_seed(4)
    w,state=_br_fixed_fixture()
    actions=candidate_actions(w,state,BRConfig(candidates=4),random.Random(7))
    graph=local_graph(w,state,actions)
    changed=replace(graph,edge_attr=torch.randn_like(graph.edge_attr)*10)
    full,no_edge,mlp=(make_ranker(v) for v in ('full_gnn','no_edge_gnn','action_mlp'))
    no_edge.load_state_dict(full.state_dict())
    assert torch.equal(no_edge(graph),no_edge(changed))
    assert not torch.allclose(full(graph),full(changed))
    changed=replace(changed,nodes=torch.randn_like(graph.nodes)*10)
    assert torch.equal(mlp(graph),mlp(changed))
    assert mlp(graph).shape==(len(actions),)
    mlp(graph).sum().backward()
    assert mlp.scorer[0].weight.grad.abs().sum()>0


def test_br_three_trainers_share_samples_and_layout_holdout():
    import random
    import torch
    from mrta_search.gnn_vns_br import BRConfig,MODEL_VARIANTS,candidate_actions,local_graph,ranking_pairs,train_ranker
    torch.set_num_threads(1)
    w,state=_br_fixed_fixture();rng=random.Random(4)
    actions=candidate_actions(w,state,BRConfig(candidates=4),rng)
    graph=local_graph(w,state,actions)
    pairs=ranking_pairs([w.evaluate(a.routes,a.bits).value for a in actions],rng)
    assert len(pairs)
    examples=[(f'train_{i}',graph,pairs) for i in range(5)]
    saved_nodes,saved_edges,saved_pairs=graph.nodes.clone(),graph.edge_attr.clone(),pairs.clone()
    records=[]
    for variant in MODEL_VARIANTS:
        model,record=train_ranker(examples,seed=20261126,epochs=2,variant=variant,progress=lambda *a:None)
        assert len(record['history'])==2 and torch.isfinite(model(graph)).all()
        records.append(record)
    assert all(r['train_layouts']==records[0]['train_layouts']
               and r['held_train_layouts']==records[0]['held_train_layouts'] for r in records)
    assert all(set(r['train_layouts']).isdisjoint(r['held_train_layouts']) for r in records)
    assert torch.equal(graph.nodes,saved_nodes) and torch.equal(graph.edge_attr,saved_edges)
    assert torch.equal(pairs,saved_pairs) and len(examples)==5


def test_br_scale_training_reuses_existing_primary_keys_without_repeating_layouts(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    original,new=object(),object()
    def entries(split,dataset='original'):
        assert split=='train'
        return iter([('train_01',original)] if dataset=='original'
                    else [('train_01',original),('scale_train_0002',new)])
    monkeypatch.setattr(runner,'iter_tasksets',entries)
    assert list(runner.training_tasksets('scale1000'))==[('train_01',original),('scale_train_0002',new)]


def test_br_validation_pairs_only_matching_feasible_runs_and_stops_test_on_failure(monkeypatch):
    import importlib,json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    names=[f'validation_{i:02d}' for i in range(1,11)]
    baseline=[dict(split='validation',layout_id=name,seed=str(seed),status='FEASIBLE',
                   cmax_s='100',runtime_s='10',termination_reason='ITERATION_LIMIT')
              for name in names for seed in runner.SEEDS]
    monkeypatch.setattr(runner,'read_baselines',lambda method:baseline)
    monkeypatch.setattr(runner,'iter_tasksets',lambda split:iter((name,None) for name in names))
    rows=[]
    for i,base in enumerate(baseline):
        for variant in runner.MODEL_VARIANTS:
            failed=variant=='full_gnn' and i<2
            rows.append(dict(base,dataset='original',trial='main',variant=variant,
                status='DEADLOCK' if failed else 'FEASIBLE',cmax_s='' if failed else '99',
                native_cmax_s='99',runtime_s='1',iterations='4',
                diagnostic=json.dumps(dict(native_feasible=True,initial_native_cmax_s=100,accepted=1))))
    admission=runner.validation_admission(rows)
    assert admission['ready_for_test'] and admission['summary']['runs']==30
    assert admission['summary']['feasible']==28 and admission['paired']['wag_vns']['pairs']==28
    assert admission['summary']['failures']=={'DEADLOCK':2}
    assert admission['paired']['wag_vns']['median_relative_cmax']==pytest.approx(-.01)
    rows[6]['status']='DEADLOCK';rows[6]['cmax_s']=''
    assert not runner.validation_admission(rows)['ready_for_test']
    monkeypatch.setattr(runner,'read_results',lambda dataset:rows)
    monkeypatch.setattr(runner,'update_scale_report',lambda **kwargs:None)
    def forbidden(*a,**kw):raise AssertionError('TEST must not load a model before admission')
    monkeypatch.setattr(runner,'load_model',forbidden)
    runner.run_experiment('test')


def test_br_resume_skips_complete_native_rows_with_multiple_evaluations(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    w,_=_br_fixed_fixture()
    rows=[]
    for source in ('original','scale1000'):
        for variant in (runner.MODEL_VARIANTS if source=='original' else ('full_gnn',)):
            for seed in (runner.SEEDS if source=='original' else runner.SEEDS[:1]):
                rows.append(dict(dataset=source,split='validation',layout_id='toy',seed=str(seed),variant=variant,
                    trial='main',coordination_calls='42',solution='saved',status='DEADLOCK'))
    monkeypatch.setattr(runner,'read_results',lambda *a:rows)
    monkeypatch.setattr(runner,'load_model',lambda *a,**k:(object(),{}))
    monkeypatch.setattr(runner,'iter_tasksets',lambda *a:[('toy',w.taskset)])
    monkeypatch.setattr(runner,'update_scale_report',lambda **k:None)
    monkeypatch.setattr(runner,'validation_admission',lambda rows:{'ready_for_test':False})
    def forbidden(*a,**k):raise AssertionError('completed native record was rerun or written again')
    monkeypatch.setattr(runner,'run_one',forbidden)
    monkeypatch.setattr(runner,'append_result',forbidden)
    runner.run_experiment('validation')


def test_br_result_resume_discards_only_the_interrupted_csv_tail(monkeypatch):
    import importlib,io
    from contextlib import contextmanager
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_vns_br')
    class MemoryFile:
        data=b''
        @property
        def parent(self):return self
        def mkdir(self,**kwargs):pass
        def exists(self):return bool(self.data)
        def read_bytes(self):return self.data
        def read_text(self,**kwargs):return self.data.decode('utf-8')
        @contextmanager
        def open(self,mode,**kwargs):
            stream=io.BytesIO(self.data) if mode=='r+b' else io.StringIO(newline='')
            yield stream
            self.data=stream.getvalue() if mode=='r+b' else self.data+stream.getvalue().encode('utf-8')
    path=MemoryFile();monkeypatch.setattr(runner,'SCALE_CSV',path)
    row={k:'1' for k in runner.SCALE_FIELDS};row['layout_id']='validation_01'
    runner.append_result(row,'scale1000')
    path.data+=b'original,validation,unfinished'
    assert len(runner.read_results('scale1000'))==1
    second=dict(row,layout_id='validation_02')
    runner.append_result(second,'scale1000')
    assert [r['layout_id'] for r in runner.read_results('scale1000')]==['validation_01','validation_02']
