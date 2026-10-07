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
    main_experimental_range_summary,
    select_init_bootstrap_devset,
    select_phase3_smokeset,
    select_phase3_smokeset_v2,
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
    # Compare identical workbook bytes under different absolute roots.
    (right / "family.xlsx").write_bytes((left / "family.xlsx").read_bytes())
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


def _selection_manifest() -> dict[str, object]:
    entries = []
    counts = (10, 20, 25, 28, 30, 31, 40, 45, 48, 50, 51, 55, 60, 65, 70,
              71, 75, 80, 85, 90, 91)
    for index, count in enumerate(counts):
        entries.append(
            {
                "instance_id": f"data/F/seed_{index:03d}::g{index:02d}_w{count:03d}",
                "relative_path": f"data/F/seed_{index:03d}.xlsx",
                "sheet_name": f"g{index:02d}_w{count:03d}",
                "actual_weld_count": count,
                "instance_geometry_hash": f"{index:064x}",
                "validation_status": "VALID",
                "duplicate_of": None,
            }
        )
    return {
        "dataset_manifest_id": "PPO_DATASET_MANIFEST_V1",
        "dataset_manifest_hash": "dataset-hash",
        "instances": entries,
    }


def test_v2_main_range_selection_retains_but_excludes_n_above_90() -> None:
    manifest = _selection_manifest()
    summary = main_experimental_range_summary(manifest)
    assert summary["valid_unique_total"] == 21
    assert summary["valid_unique_in_main_range"] == 20
    assert summary["valid_unique_out_of_main_range"] == 1
    assert summary["weld_count_distribution"]["90"] == 1
    first = select_phase3_smokeset_v2(manifest)
    second = select_phase3_smokeset_v2(manifest)
    assert first == second
    assert [entry["actual_weld_count"] for entry in first["instances"]] == [10, 50, 90]
    assert all(entry["actual_weld_count"] <= 90 for entry in first["instances"])


def test_bootstrap_devset_strata_workbooks_and_consumed_tracking_are_stable() -> None:
    manifest = _selection_manifest()
    historical = {
        "instances": [
            {"relative_path": "history/small.xlsx"},
            {"relative_path": "history/medium.xlsx"},
            {"relative_path": "history/large.xlsx"},
        ]
    }
    current = {
        "instances": [{"relative_path": "current/large_n090.xlsx"}]
    }
    first = select_init_bootstrap_devset(
        manifest, historical, additional_consumed_smokesets=(current,)
    )
    second = select_init_bootstrap_devset(
        manifest, historical, additional_consumed_smokesets=(current,)
    )
    assert first == second
    assert len(first["instances"]) == 20
    assert len({entry["relative_path"] for entry in first["instances"]}) == 20
    assert [len(group["instances"]) for group in first["strata"]] == [5, 5, 5, 5]
    assert set(first["historical_smoke_workbooks"]) <= set(
        first["development_consumed_workbooks"]
    )
    assert "current/large_n090.xlsx" in first["development_consumed_workbooks"]


def test_smoke_runner_requires_explicit_source_label_without_stale_commit() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "run_ppo_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "e4de209d872d46687af4974d030b32904192b906" not in source
    assert '"--source-commit-label"' in source
    assert "required=True" in source
