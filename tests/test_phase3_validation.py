from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrta_data.phase3_split import ROLE_ID_TEST, ROLE_TRAIN, ROLE_VALIDATION
from mrta_search.stats import SearchStats
from scripts import run_phase3_validation as validation
from scripts import run_phase3x_xsplit_gate as phase3x


def _manifests():
    dataset = validation._read_json(validation.DATASET_PATH)
    split = validation._read_json(validation.SPLIT_PATH)
    return dataset, split


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
