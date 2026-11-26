"""Fail-closed intake for historical frozen platform workbooks.

Excel support is intentionally imported only inside workbook-facing functions,
so importing :mod:`mrta_reference` or :mod:`mrta_data` does not require it.
No function in this module calls a platform generator or removes a weld row.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


DATASET_MANIFEST_ID = "PPO_DATASET_MANIFEST_V1"
CURRENT_SCHEMA_ID = "PPO_FROZEN_FAMILY_SCHEMA_V1"
LEGACY_SCHEMA_ID = "PPO_LEGACY_FROZEN_FAMILY_SCHEMA_V1"
UNRECOGNIZED_SCHEMA_ID = "UNRECOGNIZED"

RAW_COLUMNS = (
    "weld_id",
    "instance_index",
    "group_id",
    "orig_index",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "length_m",
)
CURRENT_META_COLUMNS = (
    "record_type",
    "key",
    "value",
    "data_role",
    "family_seed",
    "instance_seed",
    "requested_group_count",
    "placed_group_count",
    "full_placement",
    "actual_weld_count",
    "total_original_weld_length",
    "instance_hash",
    "source_excel",
    "source_excel_sha256",
    "generator_version",
    "spacing_m",
    "allow_rotate",
    "min_weld_length_m",
    "weld_z_m",
    "thickness_m",
    "scientific_config_sha256",
    "sheet_name",
    "generation_status",
    "failure_reason",
    "target_N",
    "actual_N",
    "geometry_hash",
    "geometry_content_hash",
    "hard_valid",
    "roundtrip_hash_match",
    "regeneration_hash_match",
)
LEGACY_META_COLUMNS = (
    "record_type",
    "key",
    "value",
    "sheet_name",
    "target_N",
    "actual_N",
    "geometry_hash",
    "geometry_content_hash",
    "requested_group_count",
    "placed_group_count",
    "hard_valid",
    "roundtrip_hash_match",
    "regeneration_hash_match",
)
COORDINATE_COLUMNS = ("x1", "y1", "z1", "x2", "y2", "z2")
WORKSPACE_X = (0.0, 20.0)
WORKSPACE_Y = (0.0, 12.0)
COORDINATE_DECIMALS = 12
Z_TOLERANCE_M = 1.0e-9
XY_TOLERANCE_M = 1.0e-12


class PlatformInstanceError(ValueError):
    """Stable fail-closed data error."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class SheetInventory:
    name: str
    row_count: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class WorkbookInspection:
    path: Path
    relative_path: str
    file_size: int
    raw_file_sha256: str
    schema_id: str
    classification: str
    sheets: tuple[SheetInventory, ...]
    family_metadata: Mapping[str, Any]
    sheet_metadata: Mapping[str, Mapping[str, Any]]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeldRow:
    weld_id: str
    source_row_ordinal: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float

    @property
    def xy_length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def length_3d(self) -> float:
        return math.dist((self.x1, self.y1, self.z1), (self.x2, self.y2, self.z2))

    @property
    def midpoint(self) -> tuple[float, float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
            (self.z1 + self.z2) / 2.0,
        )


@dataclass(frozen=True)
class PlatformInstance:
    instance_id: str
    workbook_path: Path
    relative_path: str
    sheet_name: str
    schema_id: str
    raw_file_sha256: str
    input_units: str
    coordinate_scale_to_m: float
    source_input_units: str | None
    source_row_count: int
    rows: tuple[WeldRow, ...]
    family_metadata: Mapping[str, Any]
    sheet_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    status: str
    errors: tuple[str, ...]
    z_min: float | None
    z_max: float | None


def _openpyxl():
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PlatformInstanceError(
            "EXCEL_DEPENDENCY_MISSING",
            "Platform Excel intake requires the optional openpyxl dependency",
        ) from exc
    return openpyxl


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _parse_meta_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _relative_path(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return path.name
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PlatformInstanceError(
            "WORKBOOK_OUTSIDE_PLATFORM_ROOT", f"{resolved} is outside {root.resolve()}"
        ) from exc


def _instance_id(relative_path: str, sheet_name: str) -> str:
    return f"{Path(relative_path).with_suffix('').as_posix()}::{sheet_name}"


def discover_platform_instances(platform_root: str | Path) -> tuple[Path, ...]:
    """Return every Excel workbook below ``platform_root`` in stable order."""

    root = Path(platform_root).resolve()
    if not root.is_dir():
        raise PlatformInstanceError("PLATFORM_ROOT_NOT_FOUND", str(root))
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _schema_from(meta_columns: tuple[str, ...], raw_columns: Sequence[tuple[str, ...]]) -> str:
    if meta_columns == CURRENT_META_COLUMNS and all(columns == RAW_COLUMNS for columns in raw_columns):
        return CURRENT_SCHEMA_ID
    if meta_columns == LEGACY_META_COLUMNS and all(columns == RAW_COLUMNS for columns in raw_columns):
        return LEGACY_SCHEMA_ID
    return UNRECOGNIZED_SCHEMA_ID


def inspect_platform_workbook(
    path: str | Path, *, platform_root: str | Path | None = None
) -> WorkbookInspection:
    """Inspect sheets, columns, row counts, metadata and file identity read-only."""

    workbook_path = Path(path).resolve()
    root = None if platform_root is None else Path(platform_root).resolve()
    relative = _relative_path(workbook_path, root)
    if workbook_path.suffix.lower() != ".xlsx":
        return WorkbookInspection(
            workbook_path,
            relative,
            workbook_path.stat().st_size,
            _sha256_file(workbook_path),
            UNRECOGNIZED_SCHEMA_ID,
            "UNRECOGNIZED",
            (),
            {},
            {},
            ("UNSUPPORTED_XLS_FORMAT",),
        )
    openpyxl = _openpyxl()
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=False)
    inventories: list[SheetInventory] = []
    family: dict[str, Any] = {}
    sheet_metadata: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    meta_columns: tuple[str, ...] = ()
    try:
        for worksheet in workbook.worksheets:
            row_values = worksheet.iter_rows(values_only=True)
            first = next(row_values, ())
            columns = tuple("" if value is None else str(value) for value in first)
            inventories.append(
                SheetInventory(worksheet.title, sum(1 for _ in row_values), columns)
            )
        if "META" not in workbook.sheetnames:
            errors.append("MISSING_META_SHEET")
        else:
            rows = list(workbook["META"].iter_rows(values_only=True))
            if rows:
                meta_columns = tuple(str(value) for value in rows[0] if value is not None)
                for values in rows[1:]:
                    record = dict(zip(meta_columns, values))
                    if record.get("record_type") == "family":
                        family[str(record.get("key"))] = _parse_meta_value(record.get("value"))
                    elif record.get("record_type") == "sheet" and record.get("sheet_name"):
                        sheet_metadata[str(record["sheet_name"])] = {
                            key: _parse_meta_value(value) for key, value in record.items()
                        }
            else:
                errors.append("EMPTY_META_SHEET")
        raw_columns = [item.columns for item in inventories if item.name != "META"]
        schema_id = _schema_from(meta_columns, raw_columns)
        if schema_id == UNRECOGNIZED_SCHEMA_ID:
            errors.append("UNRECOGNIZED_SCHEMA")
        expected_sheets = {"META", *sheet_metadata}
        if sheet_metadata and set(workbook.sheetnames) != expected_sheets:
            errors.append("SHEET_SET_MISMATCH")
        classification = (
            "FROZEN_PLATFORM_INSTANCE_FAMILY"
            if schema_id != UNRECOGNIZED_SCHEMA_ID
            else "UNRECOGNIZED"
        )
        return WorkbookInspection(
            workbook_path,
            relative,
            workbook_path.stat().st_size,
            _sha256_file(workbook_path),
            schema_id,
            classification,
            tuple(inventories),
            family,
            sheet_metadata,
            tuple(dict.fromkeys(errors)),
        )
    finally:
        workbook.close()


def _float(value: Any, *, field: str, location: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlatformInstanceError("NON_NUMERIC_COORDINATE", f"{location}:{field}") from exc
    if not math.isfinite(result):
        raise PlatformInstanceError("NON_FINITE_COORDINATE", f"{location}:{field}")
    return result


def load_platform_instance(
    path: str | Path,
    sheet_name: str,
    *,
    platform_root: str | Path | None = None,
    instance_id: str | None = None,
) -> PlatformInstance:
    """Load one frozen platform sheet without generation, filtering or movement."""

    inspection = inspect_platform_workbook(path, platform_root=platform_root)
    if inspection.schema_id == UNRECOGNIZED_SCHEMA_ID:
        raise PlatformInstanceError("UNRECOGNIZED_SCHEMA", inspection.relative_path)
    if sheet_name == "META" or sheet_name not in {item.name for item in inspection.sheets}:
        raise PlatformInstanceError("INSTANCE_SHEET_NOT_FOUND", sheet_name)

    openpyxl = _openpyxl()
    workbook = openpyxl.load_workbook(inspection.path, read_only=True, data_only=False)
    try:
        return _load_sheet(
            inspection,
            workbook[sheet_name],
            instance_id=instance_id,
        )
    finally:
        workbook.close()


def _load_sheet(
    inspection: WorkbookInspection,
    worksheet,
    *,
    instance_id: str | None = None,
) -> PlatformInstance:
    sheet_name = worksheet.title
    values = worksheet.iter_rows(values_only=False)
    header_cells = next(values, ())
    columns = tuple("" if cell.value is None else str(cell.value) for cell in header_cells)
    if columns != RAW_COLUMNS:
        raise PlatformInstanceError("INVALID_WELD_COLUMNS", sheet_name)
    rows: list[WeldRow] = []
    identifier = instance_id or _instance_id(inspection.relative_path, sheet_name)
    for ordinal, cells in enumerate(values):
        if any(cell.data_type == "f" for cell in cells):
            raise PlatformInstanceError("FORMULA_FORBIDDEN", f"{sheet_name}:row={ordinal + 2}")
        raw = dict(zip(columns, (cell.value for cell in cells)))
        stable_id = (
            str(raw["weld_id"]).strip()
            if raw.get("weld_id") is not None and str(raw["weld_id"]).strip()
            else f"{identifier}::weld_{ordinal:04d}"
        )
        location = f"{sheet_name}:row={ordinal + 2}"
        coordinates = {
            name: _float(raw.get(name), field=name, location=location)
            for name in COORDINATE_COLUMNS
        }
        rows.append(
            WeldRow(
                stable_id,
                ordinal,
                coordinates["x1"],
                coordinates["y1"],
                coordinates["z1"],
                coordinates["x2"],
                coordinates["y2"],
                coordinates["z2"],
            )
        )
    return PlatformInstance(
        identifier,
        inspection.path,
        inspection.relative_path,
        sheet_name,
        inspection.schema_id,
        inspection.raw_file_sha256,
        "m",
        1.0,
        (
            None
            if inspection.family_metadata.get("input_units") is None
            else str(inspection.family_metadata.get("input_units"))
        ),
        len(rows),
        tuple(rows),
        inspection.family_metadata,
        inspection.sheet_metadata.get(sheet_name, {}),
    )


def validate_platform_instance(instance: PlatformInstance) -> ValidationResult:
    errors: list[str] = []
    seen_ids: set[str] = set()
    z_values: list[float] = []
    for row in instance.rows:
        if row.weld_id in seen_ids:
            errors.append(f"DUPLICATE_WELD_ID:{row.weld_id}")
        seen_ids.add(row.weld_id)
        coordinates = (row.x1, row.y1, row.z1, row.x2, row.y2, row.z2)
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"NON_FINITE_COORDINATE:row={row.source_row_ordinal}")
            continue
        if not (WORKSPACE_X[0] <= row.x1 <= WORKSPACE_X[1]) or not (
            WORKSPACE_X[0] <= row.x2 <= WORKSPACE_X[1]
        ):
            errors.append(f"OUT_OF_BOUNDS_X:row={row.source_row_ordinal}")
        if not (WORKSPACE_Y[0] <= row.y1 <= WORKSPACE_Y[1]) or not (
            WORKSPACE_Y[0] <= row.y2 <= WORKSPACE_Y[1]
        ):
            errors.append(f"OUT_OF_BOUNDS_Y:row={row.source_row_ordinal}")
        dz = abs(row.z2 - row.z1)
        if row.xy_length <= XY_TOLERANCE_M and dz > Z_TOLERANCE_M:
            errors.append(f"UNSUPPORTED_VERTICAL_WELD:row={row.source_row_ordinal}")
        elif dz > Z_TOLERANCE_M:
            errors.append(f"UNSUPPORTED_NONPLANAR_GEOMETRY:row={row.source_row_ordinal}")
        elif row.xy_length <= XY_TOLERANCE_M:
            errors.append(f"ZERO_XY_WELD_LENGTH:row={row.source_row_ordinal}")
        z_values.extend((row.z1, row.z2))
    if z_values and max(z_values) - min(z_values) > Z_TOLERANCE_M:
        errors.append("UNSUPPORTED_NONPLANAR_GEOMETRY:MULTIPLE_Z_PLANES")
    expected = instance.sheet_metadata.get("actual_weld_count")
    if expected is None:
        expected = instance.sheet_metadata.get("actual_N")
    if expected is not None and int(expected) != len(instance.rows):
        errors.append(f"ROW_COUNT_MISMATCH:metadata={int(expected)}:actual={len(instance.rows)}")
    unique_errors = tuple(dict.fromkeys(errors))
    status = "VALID" if not unique_errors else "INVALID"
    return ValidationResult(
        status,
        unique_errors,
        min(z_values) if z_values else None,
        max(z_values) if z_values else None,
    )


def to_parent_welds(instance: PlatformInstance):
    """Project validated planar rows to the current formal XY model."""

    validation = validate_platform_instance(instance)
    if validation.status != "VALID":
        raise PlatformInstanceError("INVALID_PLATFORM_INSTANCE", ";".join(validation.errors))
    from mrta_reference.model import ParentWeld

    return tuple(
        ParentWeld(row.weld_id, (row.x1, row.y1), (row.x2, row.y2))
        for row in instance.rows
    )


def _canonical_segment(row: WeldRow) -> tuple[float, ...]:
    start = tuple(round(value, COORDINATE_DECIMALS) for value in (row.x1, row.y1, row.z1))
    end = tuple(round(value, COORDINATE_DECIMALS) for value in (row.x2, row.y2, row.z2))
    return (*min(start, end), *max(start, end))


def instance_geometry_hash(instance: PlatformInstance) -> str:
    payload = sorted(_canonical_segment(row) for row in instance.rows)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _crosses(a: float, b: float, boundary: float, epsilon: float = 1.0e-9) -> bool:
    return (a - boundary) * (b - boundary) < -(epsilon * epsilon)


def geometry_metrics(instance: PlatformInstance) -> dict[str, float | int]:
    rows = instance.rows
    xs = [value for row in rows for value in (row.x1, row.x2)]
    ys = [value for row in rows for value in (row.y1, row.y2)]
    x_coverage = (max(xs) - min(xs)) / 20.0 if xs else 0.0
    y_coverage = (max(ys) - min(ys)) / 12.0 if ys else 0.0
    midpoints = [row.midpoint for row in rows]
    if midpoints:
        mean_x = statistics.fmean(point[0] for point in midpoints)
        mean_y = statistics.fmean(point[1] for point in midpoints)
        dispersion = math.sqrt(
            statistics.fmean(
                (point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2
                for point in midpoints
            )
        ) / math.hypot(20.0, 12.0)
    else:
        dispersion = 0.0
    occupied = {
        (
            min(3, max(0, int(point[0] / 20.0 * 4))),
            min(3, max(0, int(point[1] / 12.0 * 4))),
        )
        for point in midpoints
    }
    quadrant_lengths = [0.0, 0.0, 0.0, 0.0]
    lengths: list[float] = []
    cross_y = 0
    cross_x = 0
    for row in rows:
        length = row.xy_length
        lengths.append(length)
        midpoint = row.midpoint
        quadrant = (0 if midpoint[1] >= 6.0 else 2) + (0 if midpoint[0] < 10.0 else 1)
        quadrant_lengths[quadrant] += length
        cross_y += int(_crosses(row.y1, row.y2, 6.0))
        cross_x += int(_crosses(row.x1, row.x2, 10.0))
    quadrant_mean = statistics.fmean(quadrant_lengths)
    quadrant_std = statistics.pstdev(quadrant_lengths)
    total = sum(lengths)
    upper = quadrant_lengths[0] + quadrant_lengths[1]
    lower = quadrant_lengths[2] + quadrant_lengths[3]
    left = quadrant_lengths[0] + quadrant_lengths[2]
    right = quadrant_lengths[1] + quadrant_lengths[3]
    return {
        "bbox_area_ratio": x_coverage * y_coverage,
        "x_coverage_ratio": x_coverage,
        "y_coverage_ratio": y_coverage,
        "midpoint_dispersion": dispersion,
        "grid_occupancy_ratio": len(occupied) / 16.0,
        "quadrant_length_cv": quadrant_std / quadrant_mean if quadrant_mean else 0.0,
        "upper_lower_length_imbalance": abs(upper - lower) / total if total else 0.0,
        "left_right_length_imbalance": abs(left - right) / total if total else 0.0,
        "cross_y6_count": cross_y,
        "cross_x10_count": cross_x,
        "total_weld_length_m": total,
        "weld_length_mean": statistics.fmean(lengths) if lengths else 0.0,
        "weld_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "weld_length_min": min(lengths, default=0.0),
        "weld_length_max": max(lengths, default=0.0),
    }


def formal_compatibility_metrics(instance: PlatformInstance) -> dict[str, int | str]:
    from mrta_reference.geometry import (
        blocks_for_pattern,
        generate_y_split_patterns,
        robot_is_eligible,
        whole_eligible_rails,
    )
    from mrta_reference.model import Rail, ScientificConfig

    config = ScientificConfig()
    counts: Counter[str] = Counter()
    for parent in to_parent_welds(instance):
        rails = whole_eligible_rails(parent.start, parent.end, config)
        if rails == frozenset({Rail.UPPER}):
            counts["n_whole_upper_only"] += 1
        elif rails == frozenset({Rail.LOWER}):
            counts["n_whole_lower_only"] += 1
        elif rails == frozenset({Rail.UPPER, Rail.LOWER}):
            counts["n_whole_both_rails"] += 1
        else:
            legal = [
                pattern
                for pattern in generate_y_split_patterns(parent, config)
                if all(
                    any(robot_is_eligible(block, robot, config) for robot in range(4))
                    for block in blocks_for_pattern(parent, pattern, config)
                )
            ]
            if legal:
                counts["n_mandatory_y_split"] += 1
            else:
                counts["n_no_legal_pattern"] += 1
        if _crosses(parent.start[1], parent.end[1], 6.0):
            counts["n_cross_By"] += 1
    result: dict[str, int | str] = {
        key: counts[key]
        for key in (
            "n_whole_upper_only",
            "n_whole_lower_only",
            "n_whole_both_rails",
            "n_mandatory_y_split",
            "n_no_legal_pattern",
            "n_cross_By",
        )
    }
    result["formal_compatibility"] = (
        "PASS" if counts["n_no_legal_pattern"] == 0 else "FAIL"
    )
    return result


def x_split_geometry_metrics(instance: PlatformInstance) -> dict[str, object]:
    """Return solver-independent finite X_SPLIT opportunity metadata."""
    from mrta_reference.geometry import x_split_geometry_metadata
    from mrta_reference.model import ScientificConfig

    return x_split_geometry_metadata(to_parent_welds(instance), ScientificConfig())


def _manifest_entry(instance: PlatformInstance) -> dict[str, Any]:
    validation = validate_platform_instance(instance)
    metrics = geometry_metrics(instance)
    formal: dict[str, Any] = {
        "n_whole_upper_only": 0,
        "n_whole_lower_only": 0,
        "n_whole_both_rails": 0,
        "n_mandatory_y_split": 0,
        "n_no_legal_pattern": 0,
        "n_cross_By": 0,
        "formal_compatibility": "NOT_EVALUATED",
    }
    errors = list(validation.errors)
    if validation.status == "VALID":
        formal = formal_compatibility_metrics(instance)
        if formal["formal_compatibility"] != "PASS":
            errors.append("FORMAL_NO_LEGAL_PATTERN")
    status = "VALID" if not errors else "INVALID"
    return {
        "instance_id": instance.instance_id,
        "relative_path": instance.relative_path,
        "sheet_name": instance.sheet_name,
        "raw_file_sha256": instance.raw_file_sha256,
        "schema_id": instance.schema_id,
        "input_units": instance.input_units,
        "source_input_units": instance.source_input_units,
        "coordinate_scale_to_m": instance.coordinate_scale_to_m,
        "source_row_count": instance.source_row_count,
        "actual_weld_count": len(instance.rows),
        "instance_geometry_hash": instance_geometry_hash(instance),
        "z_min": validation.z_min,
        "z_max": validation.z_max,
        "validation_status": status,
        "validation_errors": errors,
        "duplicate_of": None,
        "duplicate_reasons": [],
        **metrics,
        **formal,
    }


def _distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_dataset_manifest(platform_root: str | Path) -> dict[str, Any]:
    """Inspect and validate every recognized frozen platform sheet."""

    root = Path(platform_root).resolve()
    paths = discover_platform_instances(root)
    inspections = [inspect_platform_workbook(path, platform_root=root) for path in paths]
    inventory: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for inspection in inspections:
        inventory.append(
            {
                "relative_path": inspection.relative_path,
                "filename": inspection.path.name,
                "file_size": inspection.file_size,
                "raw_file_sha256": inspection.raw_file_sha256,
                "schema_id": inspection.schema_id,
                "classification": inspection.classification,
                "sheets": [
                    {
                        "sheet_name": sheet.name,
                        "row_count": sheet.row_count,
                        "column_names": list(sheet.columns),
                    }
                    for sheet in inspection.sheets
                ],
                "inspection_errors": list(inspection.errors),
            }
        )
        if inspection.schema_id == UNRECOGNIZED_SCHEMA_ID:
            continue
        openpyxl = _openpyxl()
        workbook = openpyxl.load_workbook(
            inspection.path, read_only=True, data_only=False
        )
        try:
            for sheet in inspection.sheets:
                if sheet.name == "META":
                    continue
                try:
                    instance = _load_sheet(inspection, workbook[sheet.name])
                    entries.append(_manifest_entry(instance))
                except PlatformInstanceError as exc:
                    identifier = _instance_id(inspection.relative_path, sheet.name)
                    entries.append(
                        {
                            "instance_id": identifier,
                            "relative_path": inspection.relative_path,
                            "sheet_name": sheet.name,
                            "raw_file_sha256": inspection.raw_file_sha256,
                            "schema_id": inspection.schema_id,
                            "input_units": "m",
                            "source_input_units": inspection.family_metadata.get("input_units"),
                            "coordinate_scale_to_m": 1.0,
                            "source_row_count": sheet.row_count,
                            "actual_weld_count": sheet.row_count,
                            "instance_geometry_hash": None,
                            "total_weld_length_m": None,
                            "z_min": None,
                            "z_max": None,
                            "validation_status": "INVALID",
                            "validation_errors": [exc.code],
                            "duplicate_of": None,
                            "duplicate_reasons": [],
                        }
                    )
        finally:
            workbook.close()

    entries.sort(key=lambda item: item["instance_id"])
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    geometry_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        raw_groups[str(entry["raw_file_sha256"])].append(entry)
        if entry["validation_status"] == "VALID" and entry.get("instance_geometry_hash"):
            geometry_groups[str(entry["instance_geometry_hash"])].append(entry)
    duplicate_raw_files = {
        digest: sorted({entry["relative_path"] for entry in group})
        for digest, group in raw_groups.items()
        if len({entry["relative_path"] for entry in group}) > 1
    }
    duplicate_geometry: dict[str, list[str]] = {}
    for digest, group in geometry_groups.items():
        ordered = sorted(group, key=lambda item: item["instance_id"])
        if len(ordered) <= 1:
            continue
        canonical = ordered[0]["instance_id"]
        duplicate_geometry[digest] = [item["instance_id"] for item in ordered]
        for item in ordered[1:]:
            item["duplicate_of"] = canonical
            item["duplicate_reasons"].append("IDENTICAL_NORMALIZED_GEOMETRY")
    for digest, paths_for_digest in duplicate_raw_files.items():
        canonical_path = min(paths_for_digest)
        for entry in raw_groups[digest]:
            if entry["relative_path"] != canonical_path:
                entry["duplicate_reasons"].append("IDENTICAL_RAW_FILE_SHA256")

    usable = [
        entry
        for entry in entries
        if entry["validation_status"] == "VALID" and entry["duplicate_of"] is None
    ]
    hash_payload = sorted(usable, key=lambda item: item["instance_id"])
    manifest_hash = hashlib.sha256(
        _canonical_json(hash_payload).encode("utf-8")
    ).hexdigest()
    invalid_reasons = Counter(
        error.split(":", 1)[0]
        for entry in entries
        if entry["validation_status"] != "VALID"
        for error in entry["validation_errors"]
    )
    schema_counts = Counter(item["schema_id"] for item in inventory)
    classifications = Counter(item["classification"] for item in inventory)
    metric_names = (
        "bbox_area_ratio",
        "x_coverage_ratio",
        "y_coverage_ratio",
        "midpoint_dispersion",
        "grid_occupancy_ratio",
        "quadrant_length_cv",
        "upper_lower_length_imbalance",
        "left_right_length_imbalance",
        "weld_length_mean",
        "weld_length_std",
        "weld_length_min",
        "weld_length_max",
    )
    return {
        "dataset_manifest_id": DATASET_MANIFEST_ID,
        "dataset_manifest_hash": manifest_hash,
        "identity_policy": (
            "SHA256(canonical JSON of VALID unique entries sorted by instance_id)"
        ),
        "ppo_root_policy": "runtime-only; manifest paths are PPO-root-relative",
        "coordinate_policy": (
            "Frozen V9 raw sheets use metre-valued x/y/z and length_m columns; "
            "META input_units describes the pre-packing source workbook."
        ),
        "projection_policy": (
            "Only single-plane z1~=z2 welds are projected to XY; no row is filtered, "
            "clipped, moved, or regenerated."
        ),
        "formal_compatibility_policy": (
            "ScientificConfig(); n_cross_By counts strict crossings of y=6.0 m"
        ),
        "inventory_summary": {
            "excel_file_count": len(inventory),
            "schema_counts": dict(sorted(schema_counts.items())),
            "classification_counts": dict(sorted(classifications.items())),
            "raw_source_assembly_count": classifications.get("RAW_SOURCE_ASSEMBLY", 0),
            "frozen_platform_family_count": classifications.get(
                "FROZEN_PLATFORM_INSTANCE_FAMILY", 0
            ),
            "unrecognized_file_count": classifications.get("UNRECOGNIZED", 0),
            "platform_instance_sheet_count": len(entries),
        },
        "validation_summary": {
            "valid_instance_count": sum(
                entry["validation_status"] == "VALID" for entry in entries
            ),
            "invalid_instance_count": sum(
                entry["validation_status"] != "VALID" for entry in entries
            ),
            "valid_unique_instance_count": len(usable),
            "invalid_reason_distribution": dict(sorted(invalid_reasons.items())),
            "weld_count_distribution": _distribution(
                [entry["actual_weld_count"] for entry in usable]
            ),
            "geometry_metric_distributions": {
                name: _distribution([entry[name] for entry in usable])
                for name in metric_names
            },
            "formal_compatibility_totals": {
                name: sum(int(entry.get(name, 0)) for entry in entries)
                for name in (
                    "n_whole_upper_only",
                    "n_whole_lower_only",
                    "n_whole_both_rails",
                    "n_mandatory_y_split",
                    "n_no_legal_pattern",
                    "n_cross_By",
                )
            },
        },
        "duplicate_analysis": {
            "identical_raw_file_sha256": duplicate_raw_files,
            "identical_normalized_geometry": duplicate_geometry,
        },
        "workbook_inventory": inventory,
        "instances": entries,
    }




def _valid_unique_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for entry in manifest["instances"]
        if entry["validation_status"] == "VALID" and entry.get("duplicate_of") is None
    ]


def main_experimental_range_summary(
    manifest: Mapping[str, Any], *, maximum_weld_count: int = 90
) -> dict[str, Any]:
    valid = _valid_unique_entries(manifest)
    in_range = [entry for entry in valid if entry["actual_weld_count"] <= maximum_weld_count]
    out_of_range = [entry for entry in valid if entry["actual_weld_count"] > maximum_weld_count]
    counts = Counter(int(entry["actual_weld_count"]) for entry in in_range)
    return {
        "maximum_weld_count": maximum_weld_count,
        "valid_unique_total": len(valid),
        "valid_unique_in_main_range": len(in_range),
        "valid_unique_out_of_main_range": len(out_of_range),
        "out_of_main_range_percentage": (
            100.0 * len(out_of_range) / len(valid) if valid else 0.0
        ),
        "weld_count_distribution": {
            str(count): counts.get(count, 0)
            for count in range(10, maximum_weld_count + 1)
        },
        "out_of_range_policy": (
            "retained as VALID manifest entries and stress instances; excluded from "
            "the Phase 3 main-range gate"
        ),
    }






def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_development_range_entries(root):
    from mrta_data.dataset_split import assert_v2_solver_access_allowed
    selected = json.loads((root / "data/manifests/development_x_split_set_v1.json").read_text(encoding="utf-8"))
    entries = [e for e in selected["instances"] if e["selection_ordinal"] in (2, 3, 5, 6, 9, 12)]
    entries.sort(key=lambda e: e["selection_ordinal"])
    roles = json.loads((root / "data/manifests/dataset_roles_v2.json").read_text(encoding="utf-8"))
    assert_v2_solver_access_allowed(roles, [e["relative_path"] for e in entries],
                                    allowed_roles=("V2_MODEL_DEVELOPMENT_CONSUMED",))
    return entries

def load_entry_parent_welds(entry, platform_root):
    return to_parent_welds(load_platform_instance(
        platform_root / entry["relative_path"], entry["sheet_name"], platform_root=platform_root,
        instance_id=entry["instance_id"]))

def read_development_selection(report):
    content=report.read_text(encoding="utf-8")
    return json.loads(content.split("<!-- SMALL30_RESULTS -->"+"\n```json\n",1)[1].split("\n```",1)[0])

def select_c2_development_entries(root):
    small=read_development_selection(root/'docs/YX_SPLIT_RANGE_05_CHANGE.md')['entries'];old=load_development_range_entries(root)
    entries=[dict(small[1],label='N35_S2',tier='N35'),dict(small[2],label='N35_S3',tier='N35'),
        dict(next(e for e in old if e['selection_ordinal']==5),label='N55_I5',tier='N55'),
        dict(next(e for e in old if e['selection_ordinal']==6),label='N55_I6',tier='N55'),
        dict(next(e for e in old if e['selection_ordinal']==9),label='N85_I9',tier='N85'),
        dict(next(e for e in old if e['selection_ordinal']==12),label='N85_I12',tier='N85')]
    from mrta_data.dataset_split import assert_v2_solver_access_allowed
    roles=json.loads((root/'data/manifests/dataset_roles_v2.json').read_text(encoding='utf-8'))
    assert_v2_solver_access_allowed(roles,[e['relative_path'] for e in entries],allowed_roles=('V2_MODEL_DEVELOPMENT_CONSUMED',))
    return entries


def external_geometry_dataset(root):
    """Read external geometry primitives without running their package initializer.

    The historical package initializer imports training and environment modules.
    A private namespace exposes only dataset/configuration/model source files.
    """
    import importlib
    import sys
    from types import ModuleType

    namespace = '_mrta_platform_geometry'
    package_dir = Path(root).resolve().parent / 'ppo' / 'src' / 'laces_weld_slim'
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(package_dir)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    return importlib.import_module(namespace + '.dataset')
