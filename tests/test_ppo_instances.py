from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrta_data.ppo_instances import (
    CURRENT_META_COLUMNS,
    CURRENT_SCHEMA_ID,
    LEGACY_META_COLUMNS,
    LEGACY_SCHEMA_ID,
    RAW_COLUMNS,
    PPOInstanceError,
    build_dataset_manifest,
    instance_geometry_hash,
    load_ppo_platform_instance,
    select_phase3_smokeset,
    validate_ppo_platform_instance,
)


def _write_family(
    path: Path,
    *,
    legacy: bool = False,
    sheets: dict[str, list[tuple[object, ...]]] | None = None,
    units: str = "mm",
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    meta = workbook.active
    meta.title = "META"
    columns = LEGACY_META_COLUMNS if legacy else CURRENT_META_COLUMNS
    meta.append(columns)
    family = {
        "dataset_protocol_version": "laces_weld_slim.frozen_multi_scale_dataset.v9.1",
        "dataset_role": "DEV_ONLY",
        "input_units": units,
        "platform_w_m": 20.0,
        "platform_h_m": 12.0,
        "weld_z_m": 0.1,
    }
    for key, value in family.items():
        record = {column: None for column in columns}
        record.update({"record_type": "family", "key": key, "value": value})
        meta.append([record[column] for column in columns])
    if sheets is None:
        sheets = {
            "g01_w003": [
                ("w0", 0, "g", 0, 1.0, 1.0, 0.1, 2.0, 1.0, 0.1, 1.0),
                ("w1", 0, "g", 1, 7.0, 5.0, 0.1, 7.0, 7.0, 0.1, 2.0),
                ("w2", 0, "g", 2, 12.0, 10.0, 0.1, 13.0, 10.0, 0.1, 1.0),
            ]
        }
    for name, rows in sheets.items():
        record = {column: None for column in columns}
        record.update(
            {
                "record_type": "sheet",
                "sheet_name": name,
                "actual_N": len(rows),
                "actual_weld_count": len(rows),
                "generation_status": "SUCCESS",
            }
        )
        meta.append([record[column] for column in columns])
        worksheet = workbook.create_sheet(name)
        worksheet.append(RAW_COLUMNS)
        for row in rows:
            worksheet.append(row)
    workbook.save(path)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [(False, CURRENT_SCHEMA_ID), (True, LEGACY_SCHEMA_ID)],
)
def test_observed_family_schemas_and_source_unit_metadata(tmp_path, legacy, expected):
    path = tmp_path / "family.xlsx"
    _write_family(path, legacy=legacy, units="mm")
    instance = load_ppo_platform_instance(path, "g01_w003", ppo_root=tmp_path)
    assert instance.schema_id == expected
    assert instance.input_units == "m"
    assert instance.source_input_units == "mm"
    assert instance.coordinate_scale_to_m == 1.0
    assert len(instance.rows) == 3


def test_stable_weld_ids_and_geometry_hash_ignore_absolute_root(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_family(left / "family.xlsx")
    _write_family(right / "family.xlsx")
    a = load_ppo_platform_instance(left / "family.xlsx", "g01_w003", ppo_root=left)
    b = load_ppo_platform_instance(right / "family.xlsx", "g01_w003", ppo_root=right)
    assert [row.weld_id for row in a.rows] == ["w0", "w1", "w2"]
    assert instance_geometry_hash(a) == instance_geometry_hash(b)
    assert build_dataset_manifest(left)["dataset_manifest_hash"] == build_dataset_manifest(right)["dataset_manifest_hash"]


@pytest.mark.parametrize(
    ("changed", "error"),
    [
        (("w0", 0, "g", 0, -0.1, 1.0, 0.1, 2.0, 1.0, 0.1, 2.1), "OUT_OF_BOUNDS_X"),
        (("w0", 0, "g", 0, 1.0, 1.0, 0.1, 1.0, 1.0, 0.2, 0.1), "UNSUPPORTED_VERTICAL_WELD"),
        (("w0", 0, "g", 0, "NaN", 1.0, 0.1, 2.0, 1.0, 0.1, 1.0), "NON_FINITE_COORDINATE"),
    ],
)
def test_fail_closed_geometry_validation(tmp_path, changed, error):
    path = tmp_path / "family.xlsx"
    rows = [
        changed,
        ("w1", 0, "g", 1, 7.0, 5.0, 0.1, 7.0, 7.0, 0.1, 2.0),
        ("w2", 0, "g", 2, 12.0, 10.0, 0.1, 13.0, 10.0, 0.1, 1.0),
    ]
    _write_family(path, sheets={"g01_w003": rows})
    if error == "NON_FINITE_COORDINATE":
        with pytest.raises(PPOInstanceError, match=error):
            load_ppo_platform_instance(path, "g01_w003", ppo_root=tmp_path)
    else:
        instance = load_ppo_platform_instance(path, "g01_w003", ppo_root=tmp_path)
        assert any(item.startswith(error) for item in validate_ppo_platform_instance(instance).errors)
        assert len(instance.rows) == 3


def test_manifest_order_hash_stability_and_duplicate_geometry(tmp_path):
    rows = {
        "g01_w003": [
            ("w0", 0, "g", 0, 1.0, 1.0, 0.1, 2.0, 1.0, 0.1, 1.0),
            ("w1", 0, "g", 1, 7.0, 5.0, 0.1, 7.0, 7.0, 0.1, 2.0),
            ("w2", 0, "g", 2, 12.0, 10.0, 0.1, 13.0, 10.0, 0.1, 1.0),
        ],
        "g02_w003": [
            ("other0", 0, "g", 0, 2.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0),
            ("other1", 0, "g", 1, 7.0, 7.0, 0.1, 7.0, 5.0, 0.1, 2.0),
            ("other2", 0, "g", 2, 13.0, 10.0, 0.1, 12.0, 10.0, 0.1, 1.0),
        ],
        "g03_w003": [
            ("x0", 0, "g", 0, 3.0, 2.0, 0.1, 4.0, 2.0, 0.1, 1.0),
            ("x1", 0, "g", 1, 8.0, 5.0, 0.1, 8.0, 7.0, 0.1, 2.0),
            ("x2", 0, "g", 2, 14.0, 9.0, 0.1, 15.0, 9.0, 0.1, 1.0),
        ],
        "g04_w003": [
            ("y0", 0, "g", 0, 4.0, 3.0, 0.1, 5.0, 3.0, 0.1, 1.0),
            ("y1", 0, "g", 1, 9.0, 5.0, 0.1, 9.0, 7.0, 0.1, 2.0),
            ("y2", 0, "g", 2, 16.0, 8.0, 0.1, 17.0, 8.0, 0.1, 1.0),
        ],
    }
    _write_family(tmp_path / "family.xlsx", sheets=rows)
    first = build_dataset_manifest(tmp_path)
    second = build_dataset_manifest(tmp_path)
    assert first["dataset_manifest_hash"] == second["dataset_manifest_hash"]
    assert [entry["instance_id"] for entry in first["instances"]] == sorted(
        entry["instance_id"] for entry in first["instances"]
    )
    duplicates = first["duplicate_analysis"]["identical_normalized_geometry"]
    assert len(duplicates) == 1
    smokeset = select_phase3_smokeset(first)
    assert len({entry["instance_id"] for entry in smokeset["instances"]}) == 3
    json.dumps(first, allow_nan=False)
