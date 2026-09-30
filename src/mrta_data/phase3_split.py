from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import statistics
from typing import Any, Mapping, Sequence


ROLE_DEVELOPMENT = "DEVELOPMENT_CONSUMED"
ROLE_TRAIN = "TRAIN_POOL"
ROLE_VALIDATION = "VALIDATION"
ROLE_ID_TEST = "ID_TEST"
ASSIGNED_ROLES = (ROLE_TRAIN, ROLE_VALIDATION, ROLE_ID_TEST)

DESCRIPTOR_FIELDS = (
    "main_range_instance_count",
    "n_min",
    "n_median",
    "n_max",
    "median_total_weld_length_m",
    "median_bbox_coverage",
    "median_x_coverage",
    "median_y_coverage",
    "median_cross_y6_count",
    "median_upper_lower_imbalance",
    "median_left_right_imbalance",
)


def _median(values: Sequence[float | int]) -> float:
    return float(statistics.median(values))


def workbook_descriptors(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate solver-independent descriptors at whole-workbook granularity."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in manifest["instances"]:
        if (
            entry.get("validation_status") == "VALID"
            and entry.get("duplicate_of") is None
            and 10 <= int(entry["actual_weld_count"]) <= 90
        ):
            grouped[str(entry["relative_path"])].append(entry)

    inventory = {
        str(item["relative_path"]): item for item in manifest["workbook_inventory"]
    }
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(inventory):
        rows = grouped.get(path, [])
        if not rows:
            raise ValueError(f"workbook has no valid main-range instance: {path}")
        n_values = [int(row["actual_weld_count"]) for row in rows]
        raw_hash = str(inventory[path]["raw_file_sha256"])
        result[path] = {
            "relative_path": path,
            "raw_file_sha256": raw_hash,
            "generation_seed_family": path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            "historical_source_role": (
                path.split("/")[1] if path.startswith("data/") else path.split("/")[0]
            ),
            "main_range_instance_count": len(rows),
            "n_min": min(n_values),
            "n_median": _median(n_values),
            "n_max": max(n_values),
            "median_total_weld_length_m": _median(
                [float(row["total_weld_length_m"]) for row in rows]
            ),
            "median_bbox_coverage": _median(
                [float(row["bbox_area_ratio"]) for row in rows]
            ),
            "median_x_coverage": _median(
                [float(row["x_coverage_ratio"]) for row in rows]
            ),
            "median_y_coverage": _median(
                [float(row["y_coverage_ratio"]) for row in rows]
            ),
            "median_cross_y6_count": _median(
                [int(row["cross_y6_count"]) for row in rows]
            ),
            "median_upper_lower_imbalance": _median(
                [float(row["upper_lower_length_imbalance"]) for row in rows]
            ),
            "median_left_right_imbalance": _median(
                [float(row["left_right_length_imbalance"]) for row in rows]
            ),
        }
    return result


def target_counts(untouched_count: int) -> dict[str, int]:
    if untouched_count < 3:
        raise ValueError("at least three untouched workbooks are required")
    validation = round(untouched_count * 0.20)
    test = round(untouched_count * 0.20)
    while validation + test >= untouched_count:
        if validation >= test:
            validation -= 1
        else:
            test -= 1
    if abs(validation - test) > 1:
        if validation > test:
            validation -= 1
            test += 1
        else:
            test -= 1
            validation += 1
    return {
        ROLE_TRAIN: untouched_count - validation - test,
        ROLE_VALIDATION: validation,
        ROLE_ID_TEST: test,
    }


def _normalized(
    descriptors: Mapping[str, Mapping[str, Any]], paths: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    columns = [
        [float(descriptors[path][field]) for path in paths]
        for field in DESCRIPTOR_FIELDS
    ]
    means = [statistics.fmean(column) for column in columns]
    scales = [statistics.pstdev(column) or 1.0 for column in columns]
    return {
        path: tuple(
            (float(descriptors[path][field]) - means[index]) / scales[index]
            for index, field in enumerate(DESCRIPTOR_FIELDS)
        )
        for path in paths
    }


def _summary(
    role_paths: Sequence[str], descriptors: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"workbook_count": len(role_paths)}
    for field in DESCRIPTOR_FIELDS:
        values = [float(descriptors[path][field]) for path in role_paths]
        result[field] = {
            "mean": statistics.fmean(values),
            "median": _median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def build_phase3_split_manifest(
    dataset_manifest: Mapping[str, Any],
    development_consumed_workbooks: Sequence[str],
) -> dict[str, Any]:
    descriptors = workbook_descriptors(dataset_manifest)
    all_paths = tuple(sorted(descriptors))
    consumed = tuple(sorted(set(development_consumed_workbooks)))
    unknown = sorted(set(consumed) - set(all_paths))
    if unknown:
        raise ValueError(f"consumed workbook missing from dataset manifest: {unknown}")
    untouched = tuple(path for path in all_paths if path not in set(consumed))
    quotas = target_counts(len(untouched))
    normalized = _normalized(descriptors, untouched)

    # Stable farthest-first order spreads extremes before central cases.  The
    # greedy score minimizes deviation of each role's normalized mean from the
    # global mean (zero), with exact quotas and deterministic hash tie-breaking.
    ordered = sorted(
        untouched,
        key=lambda path: (
            -sum(value * value for value in normalized[path]),
            descriptors[path]["raw_file_sha256"],
            path,
        ),
    )
    assigned: dict[str, list[str]] = {role: [] for role in ASSIGNED_ROLES}
    sums = {role: [0.0] * len(DESCRIPTOR_FIELDS) for role in ASSIGNED_ROLES}
    for path in ordered:
        vector = normalized[path]
        candidates = []
        for role in ASSIGNED_ROLES:
            if len(assigned[role]) >= quotas[role]:
                continue
            new_count = len(assigned[role]) + 1
            projected = [
                (sums[role][index] + vector[index]) / new_count
                for index in range(len(vector))
            ]
            balance = sum(value * value for value in projected)
            fill = new_count / quotas[role]
            candidates.append((balance, fill, ASSIGNED_ROLES.index(role), role))
        _, _, _, selected = min(candidates)
        assigned[selected].append(path)
        for index, value in enumerate(vector):
            sums[selected][index] += value

    role_by_path = {path: ROLE_DEVELOPMENT for path in consumed}
    for role, paths in assigned.items():
        role_by_path.update({path: role for path in paths})
    if len(role_by_path) != len(all_paths):
        raise AssertionError("workbook assignment is incomplete")
    if set(consumed) & set(assigned[ROLE_ID_TEST]):
        raise AssertionError("development-consumed workbook leaked into ID_TEST")

    workbooks = []
    for path in all_paths:
        entry = dict(descriptors[path])
        entry["assigned_role"] = role_by_path[path]
        entry["all_sheets_inherit_workbook_role"] = True
        entry["stress_instances_excluded_from_balance"] = sum(
            1
            for row in dataset_manifest["instances"]
            if row["relative_path"] == path
            and row.get("validation_status") == "VALID"
            and row.get("duplicate_of") is None
            and int(row["actual_weld_count"]) > 90
        )
        workbooks.append(entry)

    payload: dict[str, Any] = {
        "split_manifest_id": "PPO_PHASE3_DATA_SPLIT_V1",
        "dataset_manifest_id": dataset_manifest["dataset_manifest_id"],
        "dataset_manifest_hash": dataset_manifest["dataset_manifest_hash"],
        "split_unit": "WHOLE_WORKBOOK_AND_GENERATION_SEED_FAMILY",
        "main_range": {"minimum_weld_count": 10, "maximum_weld_count": 90},
        "stress_policy": "N>90 is LARGE_SCALE_STRESS and excluded from balancing",
        "historical_folder_role_policy": "source folder names are metadata only",
        "assignment_policy": (
            "deterministic normalized descriptor farthest-first plus greedy mean balancing; "
            "ties by raw workbook SHA-256 and path"
        ),
        "solver_independent_descriptor_fields": list(DESCRIPTOR_FIELDS),
        "counts": {
            "total_workbooks": len(all_paths),
            "development_consumed": len(consumed),
            "untouched": len(untouched),
            ROLE_TRAIN: len(assigned[ROLE_TRAIN]),
            ROLE_VALIDATION: len(assigned[ROLE_VALIDATION]),
            ROLE_ID_TEST: len(assigned[ROLE_ID_TEST]),
        },
        "development_consumed_workbooks": list(consumed),
        "role_summaries": {
            role: _summary(assigned[role], descriptors) for role in ASSIGNED_ROLES
        },
        "workbooks": workbooks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["phase3_split_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_phase3_split_manifest(payload: Mapping[str, Any]) -> None:
    workbooks = list(payload["workbooks"])
    paths = [str(item["relative_path"]) for item in workbooks]
    if len(paths) != len(set(paths)):
        raise ValueError("a workbook appears more than once")
    roles = {str(item["assigned_role"]) for item in workbooks}
    allowed = {ROLE_DEVELOPMENT, *ASSIGNED_ROLES}
    if not roles <= allowed:
        raise ValueError(f"unknown dataset roles: {sorted(roles - allowed)}")
    consumed = set(payload.get("development_consumed_workbooks", ()))
    id_test = {
        str(item["relative_path"])
        for item in workbooks
        if item["assigned_role"] == ROLE_ID_TEST
    }
    if consumed & id_test:
        raise ValueError("DEVELOPMENT_CONSUMED source role leaked into ID_TEST")
    counts = payload["counts"]
    if int(counts[ROLE_VALIDATION]) - int(counts[ROLE_ID_TEST]) not in (-1, 0, 1):
        raise ValueError("VALIDATION and ID_TEST counts differ by more than one")


def assert_solver_access_allowed(
    payload: Mapping[str, Any],
    workbook_paths: Sequence[str],
    *,
    allowed_roles: Sequence[str] = (ROLE_DEVELOPMENT, ROLE_TRAIN, ROLE_VALIDATION),
) -> None:
    """Fail before loading a workbook whose frozen Phase-3 role is disallowed."""
    role_by_path = {
        str(item["relative_path"]): str(item["assigned_role"])
        for item in payload["workbooks"]
    }
    for path in workbook_paths:
        if path not in role_by_path:
            raise ValueError(f"workbook is absent from Phase-3 split: {path}")
        if role_by_path[path] not in set(allowed_roles):
            raise PermissionError(
                f"solver access forbidden for {path}: role={role_by_path[path]}"
            )
