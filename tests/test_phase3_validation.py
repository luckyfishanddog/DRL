from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrta_data.phase3_split import (
    ROLE_ID_TEST,
    ROLE_ID_TEST_SEALED,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    ROLE_V2_DEVELOPMENT,
    ROLE_V2_TRAIN,
    ROLE_V2_VALIDATION,
    assert_v2_solver_access_allowed,
    validate_v2_role_overlay,
)
from mrta_search.stats import SearchStats
from scripts import run_phase3_validation as validation
from scripts import run_phase3x_xsplit_gate as phase3x
from scripts import run_phase3y_v2_core as phase3y


def _manifests():
    dataset = validation._read_json(validation.DATASET_PATH)
    split = validation._read_json(validation.SPLIT_PATH)
    return dataset, split


@pytest.mark.parametrize("role", (ROLE_V2_VALIDATION, ROLE_V2_TRAIN, ROLE_ID_TEST_SEALED))
def test_yr_runner_rejects_sealed_roles_before_workbook_load(monkeypatch, role):
    roles = phase3y._read_json(phase3y.V2_ROLES_PATH)
    row = next(item for item in roles["workbooks"] if item["new_v2_role"] == role)
    def forbidden(*args, **kwargs):
        pytest.fail("workbook load reached before role rejection")
    monkeypatch.setattr(phase3y, "load_smoke_parents", forbidden)
    with pytest.raises(PermissionError):
        phase3y.load_yr_parents(row, roles, Path("D:/pybullet_test/MRTA_GA/ppo"))


def test_yr_protocol_preserves_all_frozen_science_data_and_hga():
    protocol = phase3y._read_json(phase3y.YR_PROTOCOL_PATH)
    old = phase3y._read_json(phase3y.PROTOCOL_PATH)
    phase3y.verify_yr_frozen(protocol)
    assert protocol["source_phase3y_protocol_hash"] == old["phase3y_protocol_hash"]
    for field in ("scope_hash", "catalog_hashes", "methods", "instances", "seeds",
                  "checkpoints_seconds", "time_limit_seconds", "access_gate",
                  "v2_data_roles_hash", "v2_validation_set_hash"):
        assert protocol[field] == old[field]
    assert protocol["phase3yr_protocol_hash"] == phase3y._payload_hash(protocol, "phase3yr_protocol_hash")
    assert phase3y._read_json(phase3y.ARTIFACT_PATH)["decision"]["PHASE3Y_EXECUTION_STATUS"] == "FAIL"


@pytest.mark.parametrize("zero_count,passed", ((0, True), (1, True), (2, True), (3, False), (4, False), (5, False)))
def test_yr_gate_retains_maximum_two_zero_positive_instances(zero_count, passed):
    protocol = phase3y._read_json(phase3y.YR_PROTOCOL_PATH)
    old_records = phase3y._read_json(phase3y.ARTIFACT_PATH)["records"]
    positive = protocol["access_gate"]["positive_mechanism_instances"]
    records = [{**row, "x_pattern_reference_evaluated": int(row["mechanism_id"] not in positive[:zero_count])}
               for row in old_records]
    summary = phase3y.summarize_yr_records(records, protocol)
    assert summary["search_access_pass"] is passed
    assert summary["PHASE3YR_EXECUTION_STATUS"] == ("PENDING_REGRESSION" if passed else "FAIL")
    assert summary["PHASE3Z_V2_VALIDATION_AUTHORIZED"] == "NO"


def test_phase3x_runner_freezes_budget_arms_seeds_and_rejects_non_development_roles():
    _, split = _manifests()
    config = phase3x.gate_search_config()
    assert config.time_limit == 60.0
    assert config.enable_two_opt_star is False
    assert phase3x.SEEDS == (20261004, 20261005, 20261006)
    assert phase3x.ARMS == ("NO_X_CONTROL", "FINITE_X_DOMAIN")
    role_rows = {
        row["assigned_role"]: row["relative_path"]
        for row in split["workbooks"]
        if row["assigned_role"] in {ROLE_TRAIN, ROLE_VALIDATION, ROLE_ID_TEST}
    }
    assert set(role_rows) == {ROLE_TRAIN, ROLE_VALIDATION, ROLE_ID_TEST}
    for path in role_rows.values():
        with pytest.raises(PermissionError, match="DEVELOPMENT_CONSUMED only"):
            phase3x.assert_development_access(split, (path,))
    consumed = split["development_consumed_workbooks"][0]
    phase3x.assert_development_access(split, (consumed,))


def test_validation_selection_is_deterministic_unique_4_4_4_and_role_only():
    dataset, split = _manifests()
    first = validation.build_validation_set(dataset, split)
    second = validation.build_validation_set(dataset, split)
    assert first == second
    instances = first["instances"]
    assert len(instances) == 12
    assert len({item["relative_path"] for item in instances}) == 12
    assert [sum(item["tier"] == tier for item in instances) for tier in (
        "SMALL", "MEDIUM", "LARGE"
    )] == [4, 4, 4]
    role_by_path = {
        item["relative_path"]: item["assigned_role"] for item in split["workbooks"]
    }
    assert {role_by_path[item["relative_path"]] for item in instances} == {
        ROLE_VALIDATION
    }


def test_validation_set_and_protocol_hashes_are_deterministic():
    dataset, split = _manifests()
    selection = validation.build_validation_set(dataset, split)
    assert selection["validation_set_hash"] == validation._hash_payload(
        selection, "validation_set_hash"
    )
    protocol = validation.build_protocol(dataset, split, selection)
    assert protocol == validation.build_protocol(dataset, split, selection)
    assert protocol["validation_protocol_hash"] == validation._hash_payload(
        protocol, "validation_protocol_hash"
    )


def test_v2_role_overlay_and_validation_set_are_frozen_before_smoke():
    roles = phase3y._read_json(phase3y.V2_ROLES_PATH)
    selection = phase3y._read_json(phase3y.V2_VALIDATION_PATH)
    _, split = _manifests()
    validate_v2_role_overlay(roles)
    phase3y.validate_v2_roles(roles)
    phase3y.validate_v2_validation_set(selection, split)
    assert roles["v2_data_roles_hash"] == phase3y._payload_hash(
        roles, "v2_data_roles_hash"
    )
    assert selection["v2_validation_set_hash"] == phase3y._payload_hash(
        selection, "v2_validation_set_hash"
    )
    assert roles["counts"] == {
        ROLE_V2_DEVELOPMENT: 38,
        ROLE_V2_VALIDATION: 12,
        ROLE_V2_TRAIN: 31,
        ROLE_ID_TEST_SEALED: 15,
    }
    paths = {
        row["new_v2_role"]: row["relative_path"]
        for row in roles["workbooks"]
    }
    assert_v2_solver_access_allowed(
        roles, (paths[ROLE_V2_DEVELOPMENT],),
        allowed_roles=(ROLE_V2_DEVELOPMENT,),
    )
    for role in (ROLE_V2_VALIDATION, ROLE_V2_TRAIN, ROLE_ID_TEST_SEALED):
        with pytest.raises(PermissionError, match="V2 solver access forbidden"):
            assert_v2_solver_access_allowed(
                roles, (paths[role],), allowed_roles=(ROLE_V2_DEVELOPMENT,)
            )


def test_repeated_prepare_accepts_json_tuple_list_roundtrip():
    first = validation.prepare_manifests(write=False)
    second = validation.prepare_manifests(write=False)
    assert first == second


def test_seed_set_and_method_configs_are_frozen():
    assert validation.SEEDS == (
        20260928, 20260929, 20260930, 20261001, 20261002
    )
    assert validation.method_config_hashes() == {
        validation.METHOD_ALNS: (
            "53a83e8b45d91732116a64ade3bfb0bd701988c2804e2b4827339fce7130f9db"
        ),
        validation.METHOD_HGA: (
            "a1748ac615fd93270dbeb75361c14a32ca3018cca805c33361cb90309d903847"
        ),
        validation.METHOD_WAG: (
            "469c79808f2cb6c2e9d6af0c5252555ea803e0c9f500f9e8652162d071e17f67"
        ),
    }
    config = validation.validation_alns_config()
    assert config.enable_two_opt_star is False
    assert config.max_iterations == 100000
    assert config.time_limit == 60.0
    assert config.checkpoints == (5.0, 30.0, 60.0)


@pytest.mark.parametrize("forbidden_role", [ROLE_ID_TEST, ROLE_TRAIN])
def test_validation_loader_rejects_forbidden_roles_before_excel_access(
    forbidden_role, tmp_path
):
    path = f"data/{forbidden_role}/forbidden.xlsx"
    split = {
        "workbooks": [{"relative_path": path, "assigned_role": forbidden_role}]
    }
    entry = {
        "relative_path": path,
        "sheet_name": "g00",
        "instance_id": "forbidden::g00",
    }
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Excel loader must never be called")

    with pytest.raises(PermissionError):
        validation.load_validation_parents(
            split, entry, tmp_path, loader=forbidden_loader
        )
    assert called is False


def test_frozen_id_test_workbooks_are_absent_from_validation_selection():
    dataset, split = _manifests()
    selection = validation.build_validation_set(dataset, split)
    frozen_id_paths = {
        item["relative_path"]
        for item in split["workbooks"]
        if item["assigned_role"] == ROLE_ID_TEST
    }
    assert frozen_id_paths.isdisjoint(
        item["relative_path"] for item in selection["instances"]
    )


def test_checkpoint_accounting_never_backfills_from_future_improvement():
    stats = SearchStats("scope", 1)
    stats.record_best(4.0, 120.0)
    stats.record_best(31.0, 80.0)
    stats.record_best(60.1, 50.0)
    anytime = stats.anytime((5.0, 30.0, 60.0))
    assert anytime[5.0]["cmax"] == 120.0
    assert anytime[30.0]["cmax"] == 120.0
    assert anytime[60.0]["cmax"] == 80.0


def test_resume_requires_exact_key():
    provenance = {
        "validation_set_hash": "set",
        "scope_hash": "scope",
        "source_tree_hash": "tree",
    }
    key = validation._run_key(
        provenance, validation.METHOD_ALNS, "instance", validation.SEEDS[0]
    )
    record = {"run_key": key, "run_key_hash": validation._run_key_hash(key)}
    assert validation.resume_record_matches(record, key)
    changed = dict(key, source_tree_hash="different")
    assert not validation.resume_record_matches(record, changed)


def test_resume_rejects_provenance_and_protocol_mismatch(tmp_path, monkeypatch):
    artifact_path = tmp_path / "artifact.json"
    monkeypatch.setattr(validation, "ARTIFACT_PATH", artifact_path)
    protocol = {"validation_protocol_hash": "protocol-a", "methods": []}
    provenance = {"source_tree_hash": "tree-a", "validation_set_hash": "set-a"}
    artifact = validation._new_artifact(protocol, provenance)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        validation._load_or_create_artifact(
            {"validation_protocol_hash": "protocol-b", "methods": []}, provenance
        )
    with pytest.raises(ValueError, match="provenance mismatch"):
        validation._load_or_create_artifact(
            protocol, {"source_tree_hash": "tree-b", "validation_set_hash": "set-a"}
        )


def test_failed_checkpoint_has_null_not_penalty_and_median_requires_5_of_5():
    rows = [{"cmax_at_60": 100.0} for _ in range(4)] + [
        {"cmax_at_60": None}
    ]
    assert validation.checkpoint_median(rows, "cmax_at_60") is None
    assert all(row["cmax_at_60"] != 1e9 for row in rows)


def _release_records():
    return [
        {
            "final_certified": True,
            "cmax_at_60": 100.0,
            "numeric_failure_count": 0,
            "certifier_mismatch_count": 0,
            "termination_reason": "TIME_LIMIT",
        }
        for _ in range(180)
    ]


def _ratios(values):
    tiers = ("SMALL",) * 4 + ("MEDIUM",) * 4 + ("LARGE",) * 4
    return {
        "ratios": [
            {"instance_id": str(index), "tier": tier, "R_i": value}
            for index, (tier, value) in enumerate(zip(tiers, values))
        ]
    }


def test_release_rule_exact_1_10_1_15_and_8_of_12_boundaries():
    tier_summaries = [
        {"tier": tier, "median_R_i": 1.15}
        for tier in ("SMALL", "MEDIUM", "LARGE")
    ]
    passing = validation.evaluate_release(
        _release_records(), tier_summaries, _ratios([1.10] * 8 + [1.15] * 4)
    )
    assert passing["checks"]["median_R_i"] == pytest.approx(1.10)
    assert passing["checks"]["instances_R_i_at_most_1_10"] == 8
    assert passing["checks"]["tier_medians_at_most_1_15"] is True
    assert passing["DETERMINISTIC_BACKBONE_STATUS"] == "FROZEN"

    count_failure = validation.evaluate_release(
        _release_records(), tier_summaries, _ratios([1.10] * 7 + [1.15] * 5)
    )
    assert count_failure["checks"]["instances_R_i_at_most_1_10"] == 7
    assert count_failure["DETERMINISTIC_BACKBONE_STATUS"] == (
        "NEEDS_CANDIDATE_POOL_AUDIT"
    )

    tier_failure = validation.evaluate_release(
        _release_records(),
        [
            {"tier": "SMALL", "median_R_i": 1.15},
            {"tier": "MEDIUM", "median_R_i": 1.1500000001},
            {"tier": "LARGE", "median_R_i": 1.15},
        ],
        _ratios([1.10] * 12),
    )
    assert tier_failure["checks"]["tier_medians_at_most_1_15"] is False
    assert tier_failure["DETERMINISTIC_BACKBONE_STATUS"] == (
        "NEEDS_CANDIDATE_POOL_AUDIT"
    )


# Phase 3-X2 is an independent bounded diagnostic, not a production search change.
from dataclasses import replace
from types import SimpleNamespace
from scripts import run_phase3x2_x_oracle_audit as x2
from mrta_reference.model import ParentWeld, Rail, ScientificConfig, SplitKind, SplitPattern, ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V1_1, EXPERIMENTAL_X_SPLIT_SCOPE_V1
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.solution import canonicalize
from mrta_reference.certifier import certify_schedule, CertificationReport
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.initialization import InitializationResult, InitializationStatus


@pytest.fixture(scope="module")
def x2_frozen_context():
    return x2.build_context(Path("D:/pybullet_test/MRTA_GA/ppo"))


def test_x2_same_frozen_539_census_and_common_seed_hashes(x2_frozen_context):
    ctx = x2_frozen_context
    rows = ctx["census"]["patterns"]
    assert len(rows) == 539
    assert len({(r["instance_id"], r["parent_id"], r["rail"], r["t"]) for r in rows}) == 539
    replay = []
    for info in ctx["instances"].values():
        replay.extend(x2.pattern_census(info["entry"], info["parents"], ScientificConfig()))
        previous = ctx["history"]["common_seeds"][info["entry"]["instance_id"]]
        assert info["common"].solution.canonical_hash == previous["common_seed_hash"]
        assert info["common"].schedule.cmax == previous["common_seed_cmax"]
        assert info["common"].certification.certified
    assert ctx["census_hash"] == x2.digest({"census_id": "X_PATTERN_CENSUS_V1", "patterns": replay})
    assert ctx["gate"]["xsplit_gate_set_hash"] == x2.GATE_HASH
    assert EXPERIMENTAL_X_SPLIT_SCOPE_V1.scope_hash == x2.SCOPE_HASH
    assert sum(r["source"] == "MIDPOINT" for r in rows) == 424
    assert x2.build_protocol(ctx) == x2.build_protocol(ctx)


def _x2_fixture():
    config = ScientificConfig()
    parents = (
        ParentWeld("p", (4., 8.), (12., 8.)),
        ParentWeld("a", (0., 9.), (1., 9.)),
        ParentWeld("b", (2., 9.), (3., 9.)),
        ParentWeld("c", (15., 9.), (16., 9.)),
        ParentWeld("d", (17., 9.), (18., 9.)),
    )
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ("a::whole", "p::whole", "b::whole"), 1: ("c::whole", "d::whole")}, config)
    direction = optimize_directions_with_initial_feasibility(solution, config)
    schedule = reference_schedule_formal(solution, config, scope=FORMAL_SCOPE_V1_1,
                                        orientations={r: direction.directions[r] for r in range(4)})
    certificate = certify_schedule(solution, schedule, config, scope=FORMAL_SCOPE_V1_1)
    assert certificate.certified
    common = InitializationResult(InitializationStatus.SUCCESS, solution, direction.directions, schedule, certificate, ())
    row = next(r for r in x2.pattern_census({"instance_id": "fixture"}, parents, config)
               if r["parent_id"] == "p" and r["t"] == 0.5)
    return {"common": common}, row


def test_x2_enumerates_all_pairs_after_removing_whole_and_fixed_assignment(monkeypatch):
    import random
    monkeypatch.setattr(random, "randrange", lambda *a: pytest.fail("audit sampled randomly"))
    info, row = _x2_fixture()
    common = info["common"].solution
    pattern = x2.pattern_from_row(row)
    grid = x2.insertion_grid(common, pattern)
    assert grid == tuple((l, r) for l in range(3) for r in range(3))
    validator = x2.finite_x_split_validator(common.parents, ScientificConfig())
    for pos in grid:
        solution = x2.construct_insertion(common, pattern, pos, ScientificConfig(), validator)
        assert solution.routes[0].block_ids[pos[0]] == "p::0"
        assert solution.routes[1].block_ids[pos[1]] == "p::1"
        assert all("p::whole" not in r.block_ids for r in solution.routes)
        assert len([b for r in solution.routes for b in r.block_ids]) == len(common.parents) + 1
    lower = ParentWeld("low", (2., 3.), (8., 3.))
    base = canonicalize((lower,), (SplitPattern("low", SplitKind.WHOLE),), {3: ("low::whole",)}, ScientificConfig())
    low_row = x2.pattern_census({"instance_id": "lower"}, (lower,), ScientificConfig())[0]
    candidate = x2.construct_insertion(base, x2.pattern_from_row(low_row), (0, 0), ScientificConfig(),
                                     x2.finite_x_split_validator((lower,), ScientificConfig()))
    assert candidate.routes[2].block_ids == ("low::0",)
    assert candidate.routes[3].block_ids == ("low::1",)


def test_x2_all_valid_pairs_cheap_scored_caps_and_formal_access_without_mutation():
    info, row = _x2_fixture()
    before = info["common"].solution.canonical_json
    result = x2.audit_pattern(info, row)
    assert result["total_insertion_pairs"] == 9
    assert result["canonical_valid_pairs"] == result["cheap_scored_pairs"] == result["cheap_valid_pairs"] == 9
    assert result["dp_evaluated_pairs"] == 9
    assert result["direction_feasible_pairs"] > 0
    assert 1 <= result["reference_evaluated_pairs"] <= 4
    assert result["reference_status_counts"]["FEASIBLE"] > 0
    assert result["best"]["certified"]
    assert info["common"].solution.canonical_json == before
    best = x2.solution_from_payload(result["best"]["canonical_solution"], ScientificConfig())
    assert best.canonical_hash == result["best"]["solution_hash"]
    assert reference_schedule_formal(best, ScientificConfig(), scope=FORMAL_SCOPE_V1_1).status is ScheduleStatus.INFEASIBLE
    accepted = reference_schedule_formal(best, ScientificConfig(), scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
                                        orientations={r: tuple(result["best"]["directions"][r]) for r in range(4)})
    assert accepted.feasible


def test_x2_top16_top4_canonical_ties_are_order_invariant():
    candidates = [x2.InsertionCandidate(SimpleNamespace(canonical_hash=f"{i:03}"), (i, 0),
                                        (100., 20., 10., 10., f"{i:03}")) for i in range(40)]
    assert [c.positions[0] for c in x2.shortlist_cheap(list(reversed(candidates)))] == list(range(16))
    directed = [(c, SimpleNamespace(total_empty_travel=3.)) for c in candidates]
    assert [c.positions[0] for c, _ in x2.shortlist_direction(list(reversed(directed)))] == list(range(4))


def test_x2_cache_preserves_scientific_outputs_and_full_replay():
    info, row = _x2_fixture()
    cache = x2.AuditCache()
    a = x2.audit_pattern(info, row, cache)
    b = x2.audit_pattern(info, row, cache)
    c = x2.audit_pattern(info, row, x2.AuditCache(enabled=False))
    assert x2.scientific_payload(a) == x2.scientific_payload(b) == x2.scientific_payload(c)
    assert b["cache"]["direction"] > 0 and b["cache"]["reference"] > 0
    assert x2.digest(x2.scientific_payload(a)) == x2.digest(x2.scientific_payload(c))


def test_x2_execution_fails_immediately_on_certifier_mismatch():
    info, row = _x2_fixture()
    with pytest.raises(x2.AuditFailure, match="certifier FAIL"):
        x2.audit_pattern(info, row, certifier=lambda *a, **k: CertificationReport(False, ("forced mismatch",), None))


def test_x2_execution_fails_on_numeric_failure_and_scope_mismatch():
    info, row = _x2_fixture()
    def numeric(*a, **kw):
        return replace(reference_schedule_formal(*a, **kw), status=ScheduleStatus.NUMERIC_FAILURE)
    with pytest.raises(x2.AuditFailure, match="NUMERIC_FAILURE"):
        x2.audit_pattern(info, row, reference_evaluator=numeric)
    def mismatched(*a, **kw):
        return replace(reference_schedule_formal(*a, **kw), scope_hash="wrong")
    with pytest.raises(x2.AuditFailure, match="scope/policy mismatch"):
        x2.audit_pattern(info, row, reference_evaluator=mismatched)


@pytest.mark.parametrize("role", (ROLE_TRAIN, ROLE_VALIDATION, ROLE_ID_TEST))
def test_x2_data_role_rejected_even_if_consumed_list_is_inconsistent(role):
    split = {"development_consumed_workbooks": ["forbidden.xlsx"],
             "workbooks": [{"relative_path": "forbidden.xlsx", "assigned_role": role}]}
    with pytest.raises(PermissionError, match="DEVELOPMENT_CONSUMED only"):
        x2.assert_development_roles(split, ["forbidden.xlsx"])


def _potential_rows():
    return [{"instance_id": str(i), "workbook": str(i), "N": 55 if i else 25,
             "one_step_x_improvement": .01, "best_x_certified": True, "best_x_xspan": 2.} for i in range(2)]


def test_x2_stage_a_exact_threshold_two_workbooks_N_and_span_rules():
    assert x2.stage_a_rule(_potential_rows())["X_ONE_STEP_POTENTIAL"] == "PRESENT"
    for field, values in (("one_step_x_improvement", (.009999999, .01)),
                          ("workbook", ("same", "same")), ("N", (25, 25)),
                          ("best_x_xspan", (1.9999, 1.9999)),
                          ("best_x_certified", (False, True))):
        rows = _potential_rows()
        for row, value in zip(rows, values):
            row[field] = value
        assert x2.stage_a_rule(rows)["X_ONE_STEP_POTENTIAL"] == "NOT_ESTABLISHED"


def _retention_records():
    records = []
    for iid in ("a", "b"):
        for seed in phase3x.SEEDS:
            records.extend([
                {"instance_id": iid, "workbook": iid, "solver_seed": seed, "arm": "NO_X_FROM_COMMON", "cmax_at_60": 100., "x_split_parent_count": 0},
                {"instance_id": iid, "workbook": iid, "solver_seed": seed, "arm": "FINITE_X_FROM_BEST_X", "cmax_at_60": 99., "x_split_parent_count": 1},
            ])
    return records


def test_x2_stage_b_exact_retention_rule_and_four_final_cases():
    records = _retention_records()
    adequate = x2.stage_b_rule(records)
    assert adequate["X_SEARCH_RETENTION"] == "ADEQUATE"
    for r in records:
        if r["arm"] == "FINITE_X_FROM_BEST_X" and r["solver_seed"] == phase3x.SEEDS[-1]:
            r["cmax_at_60"] = 101.
    assert x2.stage_b_rule(records)["X_SEARCH_RETENTION"] == "ADEQUATE"  # exactly 2/3
    for r in records:
        if r["instance_id"] == "a" and r["solver_seed"] == phase3x.SEEDS[0] and r["arm"] == "FINITE_X_FROM_BEST_X":
            r["cmax_at_60"] = 101.
    assert x2.stage_b_rule(records)["X_SEARCH_RETENTION"] == "WEAK"
    absent = {"X_ONE_STEP_POTENTIAL": "NOT_ESTABLISHED"}
    present = {"X_ONE_STEP_POTENTIAL": "PRESENT"}
    assert x2.final_decision(absent, None)["case"] == 1
    assert x2.final_decision(present, {"X_SEARCH_RETENTION": "WEAK"})["case"] == 2
    assert x2.final_decision(present, adequate)["case"] == 3
    assert x2.final_decision(present, adequate, failed=True)["case"] == 4
    assert x2.final_decision(absent, None, complete=False)["X_DOMAIN_VALUE_STATUS"] == "UNRESOLVED"


def test_x2_stage_b_not_triggered_for_absent_potential(monkeypatch):
    monkeypatch.setattr(x2, "run_bounded_sa_oi", lambda *a, **k: pytest.fail("Stage B must not run"))
    x2.run_stage_b({"stage_a": {"X_ONE_STEP_POTENTIAL": "NOT_ESTABLISHED"}}, {})


def test_x2_resume_rejects_protocol_census_identity_and_duplicates():
    protocol = {"protocol_hash": "p", "x_pattern_census_hash": "x", "x_pattern_census": {"patterns": []}}
    a = x2.new_artifact(protocol)
    x2.validate_resume(a, protocol)
    with pytest.raises(x2.AuditFailure, match="protocol/census"):
        x2.validate_resume(dict(a, protocol_hash="changed"), protocol)
    with pytest.raises(x2.AuditFailure, match="unknown pattern"):
        x2.validate_resume(dict(a, pattern_records=[{"instance_id": "bad", "pattern_id": "bad"}]), protocol)
