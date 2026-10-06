from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mrta_baselines.common import canonical_config_hash
from mrta_baselines.hga import (
    AdaptedHGAConfig,
    METHOD_ID_V2 as METHOD_HGA,
    run_adapted_hga,
)
from mrta_baselines.wag_vns import (
    AdaptedWAGConfig,
    METHOD_ID_V2 as METHOD_WAG,
    run_adapted_wag_vns,
)
from mrta_data.phase3_split import (
    assert_v2_solver_access_allowed,
    validate_v2_role_overlay,
)
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import (
    build_legal_pattern_catalog,
    pattern_catalog_hash,
    x_split_geometry_metadata,
)
from mrta_reference.model import ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.provenance import compute_source_tree_hash
from mrta_reference.scheduler import FormalReferenceEvaluator, reference_schedule_formal
from mrta_reference.scope import (
    FORMAL_SCOPE_V1,
    FORMAL_SCOPE_V1_1,
    FORMAL_SCOPE_V2,
)
from mrta_reference.solution import official_metrics
from mrta_search.initialization import InitializationStatus, build_initial_solution
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi
from mrta_search.stats import SearchStats


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json"
PHASE3_SPLIT_PATH = ROOT / "data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json"
V2_ROLES_PATH = ROOT / "data/manifests/PPO_V2_DATA_ROLES_V1.json"
V2_VALIDATION_PATH = ROOT / "data/manifests/PPO_V2_VALIDATION_SET_V1.json"
PROTOCOL_PATH = ROOT / "data/manifests/PHASE3Y_V2_CORE_PROTOCOL_V1.json"
ARTIFACT_PATH = ROOT / "data/development/phase3y_v2_common_domain_smoke_v1.json"
X_GATE_SET_PATH = ROOT / "data/manifests/PPO_X_SPLIT_GATE_SET_V1.json"

METHOD_ALNS = "SA_OI_ALNS_V2"
METHODS = (METHOD_ALNS, METHOD_HGA, METHOD_WAG)
SEEDS = (20261005, 20261006, 20261007)
CHECKPOINTS = (5.0, 15.0, 30.0)
TIME_LIMIT = 30.0
SOURCE_COMMIT = "e8d32f02c155eaf2cae895481917421c2fe42eda"
SMOKE_ORDINALS = (2, 3, 5, 6, 9, 12)

ROLE_V2_DEVELOPMENT = "V2_MODEL_DEVELOPMENT_CONSUMED"
ROLE_V2_TRAIN = "V2_TRAIN_POOL"
ROLE_V2_VALIDATION = "V2_VALIDATION"
ROLE_ID_TEST_SEALED = "ID_TEST_SEALED"

TIERS = (
    ("SMALL", 20, 30, 25),
    ("MEDIUM", 50, 60, 55),
    ("LARGE", 80, 90, 85),
)
DESCRIPTOR_FIELDS = (
    "normalized_N",
    "total_weld_length",
    "bbox_coverage",
    "x_coverage",
    "y_coverage",
    "upper_lower_imbalance",
    "left_right_imbalance",
    "x_splittable_process_share",
    "max_x_span",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    body = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )


def _role_by_path(split: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item["relative_path"]): str(item["assigned_role"])
        for item in split["workbooks"]
    }


def _valid_entries(dataset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in dataset["instances"]
        if item.get("validation_status") == "VALID"
        and item.get("duplicate_of") is None
        and 10 <= int(item["actual_weld_count"]) <= 90
    ]


def _representative_by_workbook(
    entries: Sequence[Mapping[str, Any]],
    *,
    target: int,
    lower: int,
    upper: int,
    excluded_workbooks: set[str],
) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        path = str(entry["relative_path"])
        if path in excluded_workbooks:
            continue
        grouped.setdefault(path, []).append(entry)
    representatives = [
        min(
            rows,
            key=lambda item: (
                0 if lower <= int(item["actual_weld_count"]) <= upper else 1,
                abs(int(item["actual_weld_count"]) - target),
                str(item["instance_geometry_hash"]),
            ),
        )
        for rows in grouped.values()
    ]
    representatives.sort(
        key=lambda item: (
            0 if lower <= int(item["actual_weld_count"]) <= upper else 1,
            abs(int(item["actual_weld_count"]) - target),
            str(item["instance_geometry_hash"]),
        )
    )
    if len(representatives) < 4:
        raise ValueError(f"fewer than four candidate workbooks for target N={target}")
    # Twelve close workbook representatives provide a non-trivial diversity
    # pool while keeping proximity to the target ahead of geometry diversity.
    return representatives[: min(12, len(representatives))]


def _geometry_descriptor(
    entry: Mapping[str, Any], ppo_root: Path, config: ScientificConfig
) -> dict[str, float | int]:
    instance = load_ppo_platform_instance(
        ppo_root / str(entry["relative_path"]),
        str(entry["sheet_name"]),
        ppo_root=ppo_root,
        instance_id=str(entry["instance_id"]),
    )
    if instance.raw_file_sha256 != str(entry["raw_file_sha256"]):
        raise ValueError(f"workbook hash mismatch: {entry['relative_path']}")
    parents = to_parent_welds(instance)
    xmeta = x_split_geometry_metadata(parents, config)
    return {
        "N": int(entry["actual_weld_count"]),
        "total_weld_length": float(entry["total_weld_length_m"]),
        "bbox_coverage": float(entry["bbox_area_ratio"]),
        "x_coverage": float(entry["x_coverage_ratio"]),
        "y_coverage": float(entry["y_coverage_ratio"]),
        "upper_lower_imbalance": float(entry["upper_lower_length_imbalance"]),
        "left_right_imbalance": float(entry["left_right_length_imbalance"]),
        "cross_y6_count": int(entry["cross_y6_count"]),
        "cross_x10_count": int(entry["cross_x10_count"]),
        "x_splittable_parent_count": int(xmeta["x_splittable_parent_count"]),
        "x_split_pattern_count": int(xmeta["x_split_pattern_count"]),
        "x_splittable_process_share": float(xmeta["x_splittable_process_share"]),
        "max_x_span": float(xmeta["max_x_span"]),
        "blocking_proxy_total": float(xmeta["x_splittable_total_process_time"]),
        "x_up": float(xmeta["x_up"]),
        "x_low": float(xmeta["x_low"]),
    }


def _normalize(
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, float | int]]],
) -> dict[str, tuple[float, ...]]:
    raw_fields = (
        "N", "total_weld_length", "bbox_coverage", "x_coverage", "y_coverage",
        "upper_lower_imbalance", "left_right_imbalance",
        "x_splittable_process_share", "max_x_span",
    )
    columns = [
        [float(descriptor[field]) for _, descriptor in candidates]
        for field in raw_fields
    ]
    minima = [min(column) for column in columns]
    maxima = [max(column) for column in columns]
    result: dict[str, tuple[float, ...]] = {}
    for entry, descriptor in candidates:
        result[str(entry["instance_id"])] = tuple(
            0.0 if high == low else (float(descriptor[field]) - low) / (high - low)
            for field, low, high in zip(raw_fields, minima, maxima)
        )
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _select_four(
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, float | int]]],
    *,
    target: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, float | int], tuple[float, ...]]]:
    normalized = _normalize(candidates)
    first = min(
        candidates,
        key=lambda item: (
            abs(int(item[0]["actual_weld_count"]) - target),
            str(item[0]["instance_geometry_hash"]),
        ),
    )
    selected = [first]
    while len(selected) < 4:
        remaining = [item for item in candidates if item not in selected]
        chosen = min(
            remaining,
            key=lambda item: (
                -min(
                    _distance(
                        normalized[str(item[0]["instance_id"])],
                        normalized[str(old[0]["instance_id"])],
                    )
                    for old in selected
                ),
                abs(int(item[0]["actual_weld_count"]) - target),
                str(item[0]["instance_geometry_hash"]),
            ),
        )
        selected.append(chosen)
    return [
        (entry, descriptor, normalized[str(entry["instance_id"])])
        for entry, descriptor in selected
    ]


def build_v2_validation_set(
    dataset: Mapping[str, Any], split: Mapping[str, Any], ppo_root: Path
) -> dict[str, Any]:
    roles = _role_by_path(split)
    train_entries = [
        entry
        for entry in _valid_entries(dataset)
        if roles.get(str(entry["relative_path"])) == "TRAIN_POOL"
    ]
    selected_rows: list[dict[str, Any]] = []
    used_workbooks: set[str] = set()
    config = ScientificConfig()
    ordinal = 0
    for tier, lower, upper, target in TIERS:
        representatives = _representative_by_workbook(
            train_entries,
            target=target,
            lower=lower,
            upper=upper,
            excluded_workbooks=used_workbooks,
        )
        described = [
            (entry, _geometry_descriptor(entry, ppo_root, config))
            for entry in representatives
        ]
        for entry, descriptor, normalized in _select_four(described, target=target):
            ordinal += 1
            path = str(entry["relative_path"])
            used_workbooks.add(path)
            normalized_descriptor = dict(zip(DESCRIPTOR_FIELDS, normalized))
            selected_rows.append(
                {
                    "selection_ordinal": ordinal,
                    "tier": tier,
                    "target_N": target,
                    "relative_path": path,
                    "sheet_name": str(entry["sheet_name"]),
                    "instance_id": str(entry["instance_id"]),
                    "N": int(entry["actual_weld_count"]),
                    "instance_geometry_hash": str(entry["instance_geometry_hash"]),
                    "workbook_sha256": str(entry["raw_file_sha256"]),
                    "old_role": "TRAIN_POOL",
                    "new_v2_role": ROLE_V2_VALIDATION,
                    "geometry_descriptor": descriptor,
                    "normalized_descriptor": normalized_descriptor,
                }
            )
    payload: dict[str, Any] = {
        "validation_set_id": "PPO_V2_VALIDATION_SET_V1",
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "source_phase3_split_hash": split["phase3_split_hash"],
        "selection_policy": {
            "policy_id": "V2_GEOMETRY_CLOSE_TARGET_THEN_FARTHEST_FIRST_V1",
            "main_range": [10, 90],
            "tiers": [
                {"tier": tier, "preferred_N": [low, high], "target_N": target}
                for tier, low, high, target in TIERS
            ],
            "workbook_representative": (
                "closest preferred-range instance to tier target; tie instance_geometry_hash"
            ),
            "candidate_pool": "12 closest distinct workbook representatives per tier",
            "first": "minimum |N-target|; tie instance_geometry_hash",
            "subsequent": (
                "maximum minimum Euclidean distance in per-tier min-max normalized "
                "geometry descriptor; ties |N-target| then instance_geometry_hash"
            ),
            "descriptor_fields": list(DESCRIPTOR_FIELDS),
            "solver_results_forbidden": True,
            "one_instance_per_workbook": True,
        },
        "instances": selected_rows,
    }
    payload["v2_validation_set_hash"] = _payload_hash(
        payload, "v2_validation_set_hash"
    )
    return payload


def validate_v2_validation_set(
    payload: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    if payload.get("v2_validation_set_hash") != _payload_hash(
        payload, "v2_validation_set_hash"
    ):
        raise ValueError("v2_validation_set_hash mismatch")
    rows = list(payload.get("instances", ()))
    if len(rows) != 12:
        raise ValueError("V2 validation set must contain 12 instances")
    paths = [str(row["relative_path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("V2 validation set must use 12 distinct workbooks")
    roles = _role_by_path(split)
    if any(roles.get(path) != "TRAIN_POOL" for path in paths):
        raise ValueError("V2 validation must originate only from old TRAIN_POOL")
    counts = {tier: sum(row["tier"] == tier for row in rows) for tier, *_ in TIERS}
    if counts != {"SMALL": 4, "MEDIUM": 4, "LARGE": 4}:
        raise ValueError(f"V2 tier counts differ from 4/4/4: {counts}")


def build_v2_roles(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    validation_paths = {str(row["relative_path"]) for row in validation["instances"]}
    workbooks = []
    counts: dict[str, int] = {}
    for old in split["workbooks"]:
        old_role = str(old["assigned_role"])
        path = str(old["relative_path"])
        if old_role in {"DEVELOPMENT_CONSUMED", "VALIDATION"}:
            new_role = ROLE_V2_DEVELOPMENT
            reason = (
                "old Phase3 validation influenced V2 model decisions"
                if old_role == "VALIDATION"
                else "already development-consumed before V2"
            )
        elif old_role == "ID_TEST":
            new_role = ROLE_ID_TEST_SEALED
            reason = "original ID_TEST remains sealed and solver-inaccessible"
        elif old_role == "TRAIN_POOL" and path in validation_paths:
            new_role = ROLE_V2_VALIDATION
            reason = "solver-independent V2 geometry selection before V2 performance smoke"
        elif old_role == "TRAIN_POOL":
            new_role = ROLE_V2_TRAIN
            reason = "remaining old TRAIN_POOL; no training performed in Phase3-Y"
        else:
            raise ValueError(f"unknown old role: {old_role}")
        counts[new_role] = counts.get(new_role, 0) + 1
        workbooks.append(
            {
                "relative_path": path,
                "workbook_sha256": str(old["raw_file_sha256"]),
                "old_role": old_role,
                "new_v2_role": new_role,
                "reason": reason,
            }
        )
    payload: dict[str, Any] = {
        "roles_manifest_id": "PPO_V2_DATA_ROLES_V1",
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "source_phase3_split_hash": split["phase3_split_hash"],
        "v2_validation_set_hash": validation["v2_validation_set_hash"],
        "overlay_policy": (
            "Phase3 DEVELOPMENT_CONSUMED and VALIDATION -> V2 model-development-consumed; "
            "original ID_TEST -> sealed; 12 geometry-selected old TRAIN_POOL workbooks -> "
            "V2 validation; remaining old TRAIN_POOL -> V2 train pool"
        ),
        "counts": counts,
        "workbooks": workbooks,
    }
    payload["v2_data_roles_hash"] = _payload_hash(payload, "v2_data_roles_hash")
    return payload


def validate_v2_roles(payload: Mapping[str, Any]) -> None:
    if payload.get("v2_data_roles_hash") != _payload_hash(payload, "v2_data_roles_hash"):
        raise ValueError("v2_data_roles_hash mismatch")
    rows = list(payload.get("workbooks", ()))
    if len(rows) != 96:
        raise ValueError("V2 role overlay must contain all 96 workbooks")
    counts = payload.get("counts", {})
    expected = {
        ROLE_V2_DEVELOPMENT: 38,
        ROLE_V2_VALIDATION: 12,
        ROLE_V2_TRAIN: 31,
        ROLE_ID_TEST_SEALED: 15,
    }
    if counts != expected:
        raise ValueError(f"unexpected V2 role counts: {counts}")
    for row in rows:
        if row["old_role"] == "ID_TEST" and row["new_v2_role"] != ROLE_ID_TEST_SEALED:
            raise ValueError("old ID_TEST role was not kept sealed")


def freeze_data(ppo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _read_json(DATASET_PATH)
    split = _read_json(PHASE3_SPLIT_PATH)
    validation = build_v2_validation_set(dataset, split, ppo_root)
    validate_v2_validation_set(validation, split)
    roles = build_v2_roles(dataset, split, validation)
    validate_v2_roles(roles)
    for path, generated in ((V2_VALIDATION_PATH, validation), (V2_ROLES_PATH, roles)):
        if path.exists():
            existing = _read_json(path)
            if _canonical_json(existing) != _canonical_json(generated):
                raise ValueError(f"frozen manifest differs from deterministic rebuild: {path}")
        else:
            _write_json(path, generated)
    return roles, validation


def alns_config() -> SearchConfig:
    return SearchConfig(
        m=64,
        kdp=8,
        kref=2,
        kref_total=4,
        direction_refinement_budget=4,
        construction_budget=5,
        kinit_ref=5,
        max_iterations=100000,
        time_limit=TIME_LIMIT,
        checkpoints=CHECKPOINTS,
        enable_two_opt_star=False,
    )


def method_configs() -> dict[str, Any]:
    return {
        METHOD_ALNS: alns_config(),
        METHOD_HGA: AdaptedHGAConfig(),
        METHOD_WAG: AdaptedWAGConfig(),
    }


def method_config_hashes() -> dict[str, str]:
    return {
        method: canonical_config_hash(config)
        for method, config in method_configs().items()
    }


def smoke_entries() -> list[dict[str, Any]]:
    gate = _read_json(X_GATE_SET_PATH)
    by_ordinal = {
        int(item["selection_ordinal"]): item for item in gate["instances"]
    }
    if set(SMOKE_ORDINALS) - set(by_ordinal):
        raise ValueError("Phase3-X2 smoke ordinal is absent from frozen gate set")
    return [
        {
            **dict(by_ordinal[ordinal]),
            "mechanism_id": f"I{ordinal}",
        }
        for ordinal in SMOKE_ORDINALS
    ]


def load_smoke_parents(
    entry: Mapping[str, Any], roles: Mapping[str, Any], ppo_root: Path
):
    path = str(entry["relative_path"])
    assert_v2_solver_access_allowed(
        roles, (path,), allowed_roles=(ROLE_V2_DEVELOPMENT,)
    )
    instance = load_ppo_platform_instance(
        ppo_root / path,
        str(entry["sheet_name"]),
        ppo_root=ppo_root,
        instance_id=str(entry["instance_id"]),
    )
    if instance.raw_file_sha256 != str(entry["raw_file_sha256"]):
        raise ValueError(f"smoke workbook hash mismatch: {path}")
    return to_parent_welds(instance)


def _differential_case(
    parents: Sequence[Any], config: ScientificConfig, *, seed: int
) -> str:
    stats = SearchStats(FORMAL_SCOPE_V1_1.scope_id, seed)
    initialized = build_initial_solution(
        parents,
        config,
        stats,
        insertion_limit=8,
        construction_budget=5,
        kinit_ref=5,
        feasibility_bootstrap_budget=1,
        portfolio=True,
        reference_evaluator=FormalReferenceEvaluator(FORMAL_SCOPE_V1_1),
        scope=FORMAL_SCOPE_V1_1,
    )
    if (
        initialized.status is not InitializationStatus.SUCCESS
        or initialized.solution is None
        or initialized.directions is None
    ):
        raise RuntimeError("differential corpus initialization failed")
    if any(
        pattern.kind is SplitKind.X_SPLIT
        for pattern in initialized.solution.patterns
    ):
        raise RuntimeError("differential corpus must contain no X_SPLIT")
    directions = {
        robot: initialized.directions[robot] for robot in range(4)
    }
    old = reference_schedule_formal(
        initialized.solution,
        config,
        scope=FORMAL_SCOPE_V1_1,
        orientations=directions,
    )
    new = reference_schedule_formal(
        initialized.solution,
        config,
        scope=FORMAL_SCOPE_V2,
        orientations=directions,
    )
    if old.canonical_json() != new.canonical_json():
        raise RuntimeError("V1.1/V2 no-X schedule differential mismatch")
    if old.status is not ScheduleStatus.FEASIBLE:
        raise RuntimeError("differential corpus requires a feasible schedule")
    if official_metrics(initialized.solution, old, config) != official_metrics(
        initialized.solution, new, config
    ):
        raise RuntimeError("V1.1/V2 no-X official metrics mismatch")
    if not certify_schedule(
        initialized.solution, old, config, scope=FORMAL_SCOPE_V1_1
    ).certified:
        raise RuntimeError("V1.1 differential schedule failed certification")
    if not certify_schedule(
        initialized.solution, new, config, scope=FORMAL_SCOPE_V2
    ).certified:
        raise RuntimeError("V2 differential schedule failed certification")
    return initialized.solution.canonical_hash


def verify_core(ppo_root: Path) -> dict[str, Any]:
    if FORMAL_SCOPE_V1.scope_hash != "8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9":
        raise RuntimeError("FORMAL_SCOPE_V1 hash changed")
    if FORMAL_SCOPE_V1_1.scope_hash != "5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc":
        raise RuntimeError("FORMAL_SCOPE_V1_1 hash changed")
    FORMAL_SCOPE_V2.validate_implemented()
    roles = _read_json(V2_ROLES_PATH)
    validation = _read_json(V2_VALIDATION_PATH)
    validate_v2_roles(roles)
    validate_v2_role_overlay(roles)
    validate_v2_validation_set(validation, _read_json(PHASE3_SPLIT_PATH))
    entries = smoke_entries()
    assert_v2_solver_access_allowed(
        roles,
        tuple(str(entry["relative_path"]) for entry in entries),
        allowed_roles=(ROLE_V2_DEVELOPMENT,),
    )
    differential: list[dict[str, str]] = []
    catalog_hashes: dict[str, str] = {}
    for index, entry in enumerate(entries):
        parents = load_smoke_parents(entry, roles, ppo_root)
        catalog_hash = pattern_catalog_hash(
            parents, ScientificConfig(), FORMAL_SCOPE_V2
        )
        catalog_hashes[str(entry["instance_id"])] = catalog_hash
        differential.append(
            {
                "case": str(entry["mechanism_id"]),
                "solution_hash": _differential_case(
                    parents, ScientificConfig(), seed=20261005 + index
                ),
            }
        )
    from profile_scheduler import quality_fixture

    quality_config = ScientificConfig(
        weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0
    )
    for index, name in enumerate(
        (
            "Q1_assignment_trap",
            "Q2_route_order_trap",
            "Q3_direction_trap",
            "Q4_optional_y_split_trap",
            "Q5_interference_wait_trap",
            "Q6_lns_basin_trap",
        )
    ):
        parents, _, _ = quality_fixture(name)
        differential.append(
            {
                "case": name,
                "solution_hash": _differential_case(
                    parents, quality_config, seed=20261100 + index
                ),
            }
        )
    return {
        "scope_v1_hash": FORMAL_SCOPE_V1.scope_hash,
        "scope_v1_1_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "scope_v2_hash": FORMAL_SCOPE_V2.scope_hash,
        "v2_data_roles_hash": roles["v2_data_roles_hash"],
        "v2_validation_set_hash": validation["v2_validation_set_hash"],
        "differential_case_count": len(differential),
        "differential": differential,
        "catalog_hashes": catalog_hashes,
    }


def build_protocol(core: Mapping[str, Any]) -> dict[str, Any]:
    configs = method_configs()
    hashes = method_config_hashes()
    entries = smoke_entries()
    payload: dict[str, Any] = {
        "protocol_id": "PHASE3Y_V2_CORE_PROTOCOL_V1",
        "scope_id": FORMAL_SCOPE_V2.scope_id,
        "scope_hash": FORMAL_SCOPE_V2.scope_hash,
        "source_commit_label": SOURCE_COMMIT,
        "scientific_source_tree_hash": compute_source_tree_hash(ROOT),
        "v2_data_roles_hash": core["v2_data_roles_hash"],
        "v2_validation_set_hash": core["v2_validation_set_hash"],
        "pattern_proposal_policy_id": "PATTERN_TRANSITION_BALANCED_FAMILY_V1",
        "methods": [
            {
                "method_id": method,
                "method_config": asdict(configs[method]),
                "method_config_hash": hashes[method],
            }
            for method in METHODS
        ],
        "seeds": list(SEEDS),
        "time_limit_seconds": TIME_LIMIT,
        "checkpoints_seconds": list(CHECKPOINTS),
        "initialization_time_included": True,
        "two_opt_star": False,
        "x_oracle_seed": False,
        "run_count": len(entries) * len(METHODS) * len(SEEDS),
        "instances": entries,
        "catalog_hashes": dict(core["catalog_hashes"]),
        "access_gate": {
            "positive_mechanism_instances": ["I3", "I5", "I6", "I9", "I12"],
            "per_method_max_zero_reference_instances": 2,
            "minimum_x_reference_evaluations_per_positive_instance_across_three_seeds": 1,
        },
        "completion_gate": {
            "complete_certified_runs": 54,
            "numeric_failure_count": 0,
            "certifier_mismatch_count": 0,
            "catalog_hashes_equal_across_methods": True,
        },
        "data_prohibitions": [
            ROLE_V2_VALIDATION,
            ROLE_V2_TRAIN,
            ROLE_ID_TEST_SEALED,
        ],
    }
    payload["phase3y_protocol_hash"] = _payload_hash(
        payload, "phase3y_protocol_hash"
    )
    return payload


def freeze_protocol(ppo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    core = verify_core(ppo_root)
    generated = build_protocol(core)
    if PROTOCOL_PATH.exists():
        existing = _read_json(PROTOCOL_PATH)
        if _canonical_json(existing) != _canonical_json(generated):
            raise ValueError("frozen Phase3-Y protocol differs from deterministic rebuild")
        protocol = existing
    else:
        _write_json(PROTOCOL_PATH, generated)
        protocol = generated
    return core, protocol


def _final_pattern_fields(solution: Any | None) -> dict[str, Any]:
    if solution is None:
        return {
            "final_x_parent_count": 0,
            "final_x_pattern_ids": [],
            "final_y_parent_count": 0,
            "final_y_pattern_ids": [],
        }
    x = [
        pattern.pattern_id
        for pattern in solution.patterns
        if pattern.kind is SplitKind.X_SPLIT
    ]
    y = [
        pattern.pattern_id
        for pattern in solution.patterns
        if pattern.kind is SplitKind.Y_SPLIT
    ]
    return {
        "final_x_parent_count": len(x),
        "final_x_pattern_ids": sorted(x),
        "final_y_parent_count": len(y),
        "final_y_pattern_ids": sorted(y),
    }


def _run_key(
    protocol: Mapping[str, Any],
    method_id: str,
    entry: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "phase3y_protocol_hash": protocol["phase3y_protocol_hash"],
        "scope_hash": FORMAL_SCOPE_V2.scope_hash,
        "catalog_hash": protocol["catalog_hashes"][str(entry["instance_id"])],
        "method_id": method_id,
        "method_config_hash": method_config_hashes()[method_id],
        "instance_id": str(entry["instance_id"]),
        "solver_seed": seed,
    }


def _common_record_fields(
    protocol: Mapping[str, Any],
    method_id: str,
    entry: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    key = _run_key(protocol, method_id, entry, seed)
    return {
        "run_key": key,
        "run_key_hash": hashlib.sha256(
            _canonical_json(key).encode("utf-8")
        ).hexdigest(),
        "method_id": method_id,
        "method_config_hash": key["method_config_hash"],
        "solver_seed": seed,
        "mechanism_id": entry["mechanism_id"],
        "instance_id": entry["instance_id"],
        "workbook": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "N": entry["N"],
        "scope_id": FORMAL_SCOPE_V2.scope_id,
        "scope_hash": FORMAL_SCOPE_V2.scope_hash,
        "pattern_catalog_hash": key["catalog_hash"],
    }


def _alns_record(
    result: Any,
    *,
    protocol: Mapping[str, Any],
    entry: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    stats = result.stats
    solution = result.best_solution
    record = {
        **_common_record_fields(protocol, METHOD_ALNS, entry, seed),
        "termination_reason": result.termination_reason,
        "runtime": result.runtime,
        "overshoot": stats.overshoot,
        "iterations": stats.iterations,
        "cmax_at_5": result.anytime.get(5.0, {}).get("cmax"),
        "cmax_at_15": result.anytime.get(15.0, {}).get("cmax"),
        "cmax_at_30": result.anytime.get(30.0, {}).get("cmax"),
        "final_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "final_schedule_status": (
            None if result.best_schedule is None else result.best_schedule.status.value
        ),
        "final_certified": bool(
            result.final_certification and result.final_certification.certified
        ),
        "numeric_failure_count": int(stats.n_numeric_failure)
        + int(result.status.value == "NUMERIC_FAILURE"),
        "certifier_mismatch_count": int(
            result.best_schedule is not None
            and result.best_schedule.status is ScheduleStatus.FEASIBLE
            and not (
                result.final_certification and result.final_certification.certified
            )
        ),
        "legal_x_pattern_count": sum(
            pattern.kind is SplitKind.X_SPLIT
            for patterns in build_legal_pattern_catalog(
                result.initialization.solution.parents, ScientificConfig(), FORMAL_SCOPE_V2
            ).values()
            for pattern in patterns
        ) if result.initialization.solution is not None else 0,
        "x_pattern_proposals": stats.x_pattern_candidates_generated,
        "x_pattern_constructed": stats.x_pattern_candidates_cheap_feasible,
        "x_pattern_direction_evaluated": stats.x_pattern_candidates_c3,
        "x_pattern_reference_evaluated": stats.x_pattern_candidates_reference_evaluated,
        "x_pattern_certified": stats.x_pattern_candidates_certified,
        "x_pattern_accepted": stats.x_pattern_candidates_accepted,
        "x_pattern_global_best_updates": stats.x_pattern_global_best_updates,
        "y_pattern_proposals": stats.y_pattern_candidates_generated,
        "y_pattern_constructed": stats.y_pattern_candidates_cheap_feasible,
        "y_pattern_direction_evaluated": stats.y_pattern_candidates_c3,
        "y_pattern_reference_evaluated": stats.y_pattern_candidates_reference_evaluated,
        "y_pattern_certified": stats.y_pattern_candidates_certified,
        "y_pattern_accepted": stats.y_pattern_candidates_accepted,
        "y_pattern_global_best_updates": stats.y_pattern_global_best_updates,
        "two_opt_star_enabled": False,
        "x_oracle_seed_used": False,
        **_final_pattern_fields(solution),
    }
    return record


def _baseline_record(
    result: Any,
    *,
    protocol: Mapping[str, Any],
    entry: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    accounting = result.accounting
    record = {
        **_common_record_fields(protocol, result.method_id, entry, seed),
        "termination_reason": result.termination_reason,
        "runtime": result.actual_runtime,
        "overshoot": result.overshoot,
        "iterations": result.iterations,
        "cmax_at_5": result.checkpoints.get(5.0),
        "cmax_at_15": result.checkpoints.get(15.0),
        "cmax_at_30": result.checkpoints.get(30.0),
        "final_cmax": None if result.metrics is None else result.metrics.cmax,
        "final_schedule_status": (
            None if result.schedule is None else result.schedule.status.value
        ),
        "final_certified": result.final_certified,
        "numeric_failure_count": int(accounting["numeric_failure"]),
        "certifier_mismatch_count": int(
            result.schedule is not None
            and result.schedule.status is ScheduleStatus.FEASIBLE
            and not result.final_certified
        ),
        "legal_x_pattern_count": int(accounting["legal_x_pattern_count"]),
        "x_pattern_proposals": int(accounting["x_pattern_proposals"]),
        "x_pattern_constructed": int(accounting["x_pattern_constructed"]),
        "x_pattern_direction_evaluated": int(accounting["x_pattern_direction_evaluated"]),
        "x_pattern_reference_evaluated": int(accounting["x_pattern_reference_evaluated"]),
        "x_pattern_certified": int(accounting["x_pattern_certified"]),
        "x_pattern_accepted": int(accounting["x_pattern_accepted"]),
        "x_pattern_global_best_updates": int(accounting["x_pattern_global_best_updates"]),
        "y_pattern_proposals": int(accounting["y_pattern_proposals"]),
        "y_pattern_constructed": int(accounting["y_pattern_constructed"]),
        "y_pattern_direction_evaluated": int(accounting["y_pattern_direction_evaluated"]),
        "y_pattern_reference_evaluated": int(accounting["y_pattern_reference_evaluated"]),
        "y_pattern_certified": int(accounting["y_pattern_certified"]),
        "y_pattern_accepted": int(accounting["y_pattern_accepted"]),
        "y_pattern_global_best_updates": int(accounting["y_pattern_global_best_updates"]),
        "two_opt_star_enabled": False,
        "x_oracle_seed_used": False,
        "diagnostics": list(result.diagnostics),
        **_final_pattern_fields(result.solution),
    }
    return record


def run_method(
    method_id: str,
    parents: Sequence[Any],
    *,
    protocol: Mapping[str, Any],
    entry: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    if method_id == METHOD_ALNS:
        result = run_bounded_sa_oi(
            parents,
            ScientificConfig(),
            alns_config(),
            seed=seed,
            scope=FORMAL_SCOPE_V2,
            source_commit=SOURCE_COMMIT,
            allow_unverified_source=True,
            formal_result=False,
            enable_x_split=True,
        )
        record = _alns_record(result, protocol=protocol, entry=entry, seed=seed)
    elif method_id == METHOD_HGA:
        result = run_adapted_hga(
            parents,
            ScientificConfig(),
            AdaptedHGAConfig(),
            seed=seed,
            time_limit=TIME_LIMIT,
            scope=FORMAL_SCOPE_V2,
            checkpoints=CHECKPOINTS,
            method_id=METHOD_HGA,
        )
        record = _baseline_record(
            result, protocol=protocol, entry=entry, seed=seed
        )
    elif method_id == METHOD_WAG:
        result = run_adapted_wag_vns(
            parents,
            ScientificConfig(),
            AdaptedWAGConfig(),
            seed=seed,
            time_limit=TIME_LIMIT,
            scope=FORMAL_SCOPE_V2,
            checkpoints=CHECKPOINTS,
            method_id=METHOD_WAG,
        )
        record = _baseline_record(
            result, protocol=protocol, entry=entry, seed=seed
        )
    else:
        raise ValueError(f"unknown V2 method: {method_id}")
    if record["method_config_hash"] != method_config_hashes()[method_id]:
        raise RuntimeError("method config hash differs from frozen protocol")
    return record


def summarize_records(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    expected = int(protocol["run_count"])
    unique = {str(row["run_key_hash"]) for row in records}
    complete = len(records) == expected and len(unique) == expected
    numeric = sum(int(row["numeric_failure_count"]) for row in records)
    mismatches = sum(int(row["certifier_mismatch_count"]) for row in records)
    certified = sum(bool(row["final_certified"]) for row in records)
    positive = tuple(protocol["access_gate"]["positive_mechanism_instances"])
    access: dict[str, Any] = {}
    access_pass = True
    max_zero = int(
        protocol["access_gate"]["per_method_max_zero_reference_instances"]
    )
    for method in METHODS:
        zero = []
        by_instance = {}
        for mechanism in positive:
            total = sum(
                int(row["x_pattern_reference_evaluated"])
                for row in records
                if row["method_id"] == method
                and row["mechanism_id"] == mechanism
            )
            by_instance[mechanism] = total
            if total == 0:
                zero.append(mechanism)
        method_pass = len(zero) <= max_zero
        access_pass &= method_pass
        access[method] = {
            "x_reference_evaluations": by_instance,
            "zero_reference_instances": zero,
            "pass": method_pass,
        }
    catalog_pass = all(
        row["pattern_catalog_hash"]
        == protocol["catalog_hashes"][str(row["instance_id"])]
        for row in records
    )
    passed = (
        complete
        and certified == expected
        and numeric == 0
        and mismatches == 0
        and access_pass
        and catalog_pass
    )
    return {
        "PHASE3Y_EXECUTION_STATUS": "PASS" if passed else "FAIL",
        "FORMAL_SCOPE_V2_STATUS": "CLOSED" if passed else "OPEN",
        "COMMON_DOMAIN_V2_STATUS": "PASS" if passed else "FAIL",
        "completed_runs": len(records),
        "expected_runs": expected,
        "certified_final_schedules": certified,
        "numeric_failure_count": numeric,
        "certifier_mismatch_count": mismatches,
        "catalog_equivalence_pass": catalog_pass,
        "search_access": access,
        "search_access_pass": access_pass,
        "V2_VALIDATION_SET_STATUS": "FROZEN",
        "V2_TRAIN_POOL_WORKBOOK_COUNT": 31,
        "PHASE3Z_V2_VALIDATION_AUTHORIZED": "YES" if passed else "NO",
        "ID_TEST_STATUS": "SEALED",
    }


def run_smoke(ppo_root: Path) -> dict[str, Any]:
    core, protocol = freeze_protocol(ppo_root)
    if protocol["scientific_source_tree_hash"] != compute_source_tree_hash(ROOT):
        raise RuntimeError("scientific source changed after Phase3-Y protocol freeze")
    roles = _read_json(V2_ROLES_PATH)
    validate_v2_roles(roles)
    if ARTIFACT_PATH.exists():
        artifact = _read_json(ARTIFACT_PATH)
        if artifact.get("phase3y_protocol_hash") != protocol["phase3y_protocol_hash"]:
            raise ValueError("existing Phase3-Y artifact uses a different protocol")
    else:
        artifact = {
            "artifact_id": "PHASE3Y_V2_COMMON_DOMAIN_SMOKE_V1",
            "phase3y_protocol_hash": protocol["phase3y_protocol_hash"],
            "core_verification": core,
            "records": [],
            "decision": None,
        }
        _write_json(ARTIFACT_PATH, artifact)
    records = list(artifact["records"])
    completed = {str(row["run_key_hash"]) for row in records}
    for entry in protocol["instances"]:
        parents = load_smoke_parents(entry, roles, ppo_root)
        actual_catalog = pattern_catalog_hash(
            parents, ScientificConfig(), FORMAL_SCOPE_V2
        )
        expected_catalog = protocol["catalog_hashes"][str(entry["instance_id"])]
        if actual_catalog != expected_catalog:
            raise RuntimeError("pattern catalog changed after protocol freeze")
        for method in METHODS:
            for seed in SEEDS:
                key = _run_key(protocol, method, entry, seed)
                key_hash = hashlib.sha256(
                    _canonical_json(key).encode("utf-8")
                ).hexdigest()
                if key_hash in completed:
                    continue
                record = run_method(
                    method,
                    parents,
                    protocol=protocol,
                    entry=entry,
                    seed=seed,
                )
                records.append(record)
                completed.add(key_hash)
                artifact["records"] = records
                artifact["decision"] = summarize_records(records, protocol)
                _write_json(ARTIFACT_PATH, artifact)
                print(
                    f"{len(records)}/{protocol['run_count']} "
                    f"{entry['mechanism_id']} {method} seed={seed} "
                    f"certified={record['final_certified']} "
                    f"x_ref={record['x_pattern_reference_evaluated']}",
                    flush=True,
                )
    artifact["decision"] = summarize_records(records, protocol)
    _write_json(ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase3-Y FORMAL_SCOPE_V2 core closure")
    parser.add_argument(
        "--ppo-root", type=Path, default=Path("D:/pybullet_test/MRTA_GA/ppo")
    )
    parser.add_argument("--freeze-data", action="store_true")
    parser.add_argument("--verify-core", action="store_true")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    selected = sum((args.freeze_data, args.verify_core, args.freeze_protocol, args.run))
    if selected != 1:
        parser.error("select exactly one action")
    if args.freeze_data:
        roles, validation = freeze_data(args.ppo_root)
        output = {
            "v2_data_roles_hash": roles["v2_data_roles_hash"],
            "v2_validation_set_hash": validation["v2_validation_set_hash"],
            "counts": roles["counts"],
        }
    elif args.verify_core:
        output = verify_core(args.ppo_root)
    elif args.freeze_protocol:
        core, protocol = freeze_protocol(args.ppo_root)
        output = {
            "core": core,
            "phase3y_protocol_hash": protocol["phase3y_protocol_hash"],
        }
    else:
        artifact = run_smoke(args.ppo_root)
        output = artifact["decision"]
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
