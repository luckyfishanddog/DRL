from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mrta_data.phase3_split import (
    ROLE_DEVELOPMENT,
    ROLE_ID_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    build_phase3_split_manifest,
    assert_solver_access_allowed,
    validate_phase3_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _build():
    dataset = json.loads(
        (ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json").read_text(encoding="utf-8")
    )
    consumed = json.loads(
        (ROOT / "data/manifests/PPO_INIT_BOOTSTRAP_DEVSET_V1.json").read_text(
            encoding="utf-8"
        )
    )["development_consumed_workbooks"]
    return build_phase3_split_manifest(dataset, consumed)


def test_phase3_split_is_deterministic_balanced_and_workbook_isolated():
    first = _build()
    second = _build()
    assert first == second
    assert first["counts"] == {
        "total_workbooks": 96,
        "development_consumed": 23,
        "untouched": 73,
        ROLE_TRAIN: 43,
        ROLE_VALIDATION: 15,
        ROLE_ID_TEST: 15,
    }
    validate_phase3_split_manifest(first)
    paths = [item["relative_path"] for item in first["workbooks"]]
    assert len(paths) == len(set(paths)) == 96
    assert all(item["all_sheets_inherit_workbook_role"] for item in first["workbooks"])
    consumed = set(first["development_consumed_workbooks"])
    test_paths = {
        item["relative_path"]
        for item in first["workbooks"]
        if item["assigned_role"] == ROLE_ID_TEST
    }
    assert not consumed & test_paths
    assert {
        item["assigned_role"]
        for item in first["workbooks"]
        if item["relative_path"] in consumed
    } == {ROLE_DEVELOPMENT}


def test_historical_directory_name_does_not_override_phase3_role():
    payload = _build()
    historical_id = [
        item
        for item in payload["workbooks"]
        if item["historical_source_role"] == "ID_TEST"
        and item["assigned_role"] != ROLE_DEVELOPMENT
    ]
    assert historical_id
    assert any(item["assigned_role"] != ROLE_ID_TEST for item in historical_id)


def test_split_validator_rejects_workbook_leakage_and_duplicates():
    payload = _build()
    leaked = copy.deepcopy(payload)
    consumed_path = leaked["development_consumed_workbooks"][0]
    next(
        item for item in leaked["workbooks"] if item["relative_path"] == consumed_path
    )["assigned_role"] = ROLE_ID_TEST
    with pytest.raises(ValueError, match="leaked"):
        validate_phase3_split_manifest(leaked)

    duplicate = copy.deepcopy(payload)
    duplicate["workbooks"].append(copy.deepcopy(duplicate["workbooks"][0]))
    with pytest.raises(ValueError, match="more than once"):
        validate_phase3_split_manifest(duplicate)


def test_id_test_solver_access_is_rejected_before_workbook_loading():
    payload = _build()
    id_test = next(
        item["relative_path"]
        for item in payload["workbooks"]
        if item["assigned_role"] == ROLE_ID_TEST
    )
    with pytest.raises(PermissionError, match="solver access forbidden"):
        assert_solver_access_allowed(payload, (id_test,))

