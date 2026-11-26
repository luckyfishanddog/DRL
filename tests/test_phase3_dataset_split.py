from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mrta_data.dataset_split import (
    ROLE_DEVELOPMENT,
    ROLE_ID_TEST,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    build_dataset_split_manifest,
    build_development_diagnostic_set,
    assert_solver_access_allowed,
    validate_development_diagnostic_set,
    validate_dataset_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _build():
    dataset = json.loads(
        (ROOT / "data/manifests/platform_dataset_manifest_v1.json").read_text(encoding="utf-8")
    )
    consumed = json.loads(
        (ROOT / "data/manifests/development_consumed_set_v1.json").read_text(
            encoding="utf-8"
        )
    )["development_consumed_workbooks"]
    return build_dataset_split_manifest(dataset, consumed)


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
    validate_dataset_split_manifest(first)
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
        validate_dataset_split_manifest(leaked)

    duplicate = copy.deepcopy(payload)
    duplicate["workbooks"].append(copy.deepcopy(duplicate["workbooks"][0]))
    with pytest.raises(ValueError, match="more than once"):
        validate_dataset_split_manifest(duplicate)


def test_id_test_solver_access_is_rejected_before_workbook_loading():
    payload = _build()
    id_test = next(
        item["relative_path"]
        for item in payload["workbooks"]
        if item["assigned_role"] == ROLE_ID_TEST
    )
    with pytest.raises(PermissionError, match="solver access forbidden"):
        assert_solver_access_allowed(payload, (id_test,))


def test_phase3_diagnostic_set_is_solver_independent_deterministic_and_consumed_only():
    dataset = json.loads(
        (ROOT / "data/manifests/platform_dataset_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    split = _build()
    first = build_development_diagnostic_set(dataset, split)
    second = build_development_diagnostic_set(dataset, split)
    assert first == second
    validate_development_diagnostic_set(first, split)
    assert len(first["instances"]) == 6
    assert len({item["relative_path"] for item in first["instances"]}) == 6
    assert {item["phase3_role"] for item in first["instances"]} == {
        ROLE_DEVELOPMENT
    }
    assert [item["tier"] for item in first["instances"]].count("small") == 2
    assert [item["tier"] for item in first["instances"]].count("medium") == 2
    assert [item["tier"] for item in first["instances"]].count("large") == 2


def _sample_platform_instance():
    from mrta_data.platform_instances import inspect_platform_workbook, load_platform_instance

    root = ROOT / 'data/local/platform_samples'
    workbook = root / 'data/DEV_ONLY/seed_000202.xlsx'
    inspection = inspect_platform_workbook(workbook, platform_root=root)
    sheet = next(item.name for item in inspection.sheets if item.name != 'META')
    return load_platform_instance(workbook, sheet, platform_root=root), inspection


def test_platform_reader_preserves_frozen_coordinates_units_rows_and_geometry_hash():
    from dataclasses import replace
    from mrta_data.platform_instances import instance_geometry_hash, to_parent_welds, validate_platform_instance

    instance, inspection = _sample_platform_instance()
    assert instance.input_units == 'm' and instance.coordinate_scale_to_m == 1.0
    assert instance.raw_file_sha256 == inspection.raw_file_sha256
    assert instance.source_row_count == len(instance.rows) > 0
    assert validate_platform_instance(instance).status == 'VALID'
    parents = to_parent_welds(instance)
    assert [(p.parent_id, p.start, p.end) for p in parents] == [
        (r.weld_id, (r.x1, r.y1), (r.x2, r.y2)) for r in instance.rows
    ]
    reversed_rows = tuple(
        replace(r, x1=r.x2, y1=r.y2, z1=r.z2, x2=r.x1, y2=r.y1, z2=r.z1)
        for r in reversed(instance.rows)
    )
    assert instance_geometry_hash(instance) == instance_geometry_hash(replace(instance, rows=reversed_rows))


def test_platform_reader_rejects_nonplanar_geometry_without_filtering_rows():
    from dataclasses import replace
    from mrta_data.platform_instances import PlatformInstanceError, to_parent_welds, validate_platform_instance

    instance, _ = _sample_platform_instance()
    first = replace(instance.rows[0], z2=instance.rows[0].z1 + 0.1)
    invalid = replace(instance, rows=(first, *instance.rows[1:]))
    result = validate_platform_instance(invalid)
    assert result.status == 'INVALID'
    assert any(error.startswith('UNSUPPORTED_NONPLANAR_GEOMETRY') for error in result.errors)
    with pytest.raises(PlatformInstanceError, match='INVALID_PLATFORM_INSTANCE'):
        to_parent_welds(invalid)
    assert len(invalid.rows) == len(instance.rows)


def test_neutral_development_selection_and_v2_sealed_roles_remain_enforced():
    from mrta_data.platform_instances import load_development_range_entries
    from mrta_data.dataset_split import assert_v2_solver_access_allowed

    entries = load_development_range_entries(ROOT)
    assert [e['selection_ordinal'] for e in entries] == [2, 3, 5, 6, 9, 12]
    roles = json.loads((ROOT / 'data/manifests/dataset_roles_v2.json').read_text(encoding='utf-8'))
    assert_v2_solver_access_allowed(roles, [e['relative_path'] for e in entries])
    for role in ('V2_VALIDATION', 'ID_TEST_SEALED'):
        workbook = next(r['relative_path'] for r in roles['workbooks'] if r['new_v2_role'] == role)
        with pytest.raises(PermissionError, match='access forbidden'):
            assert_v2_solver_access_allowed(roles, [workbook])


def test_external_geometry_intake_does_not_import_training_or_environment_modules():
    import sys
    from mrta_data.platform_instances import external_geometry_dataset

    if not (ROOT.parent / 'ppo/src/laces_weld_slim/dataset.py').is_file():
        pytest.skip('External historical geometry source is unavailable')
    module = external_geometry_dataset(ROOT)
    assert callable(module.load_seed_family_workbook)
    imported = {name.rsplit('.', 1)[-1] for name in sys.modules if name.startswith('_mrta_platform_geometry.')}
    assert {'dataset', 'configuration', 'model'} <= imported
    assert not imported & {'ppo_env', 'macro_env', 'macro_ppo', 'candidate_policy'}
