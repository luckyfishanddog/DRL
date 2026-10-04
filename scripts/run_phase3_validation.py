from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from mrta_baselines.common import canonical_config_hash, solution_telemetry
from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns
from mrta_data.phase3_split import (
    ROLE_VALIDATION,
    assert_solver_access_allowed,
    validate_phase3_split_manifest,
)
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.model import ScheduleStatus, ScientificConfig
from mrta_reference.provenance import REPOSITORY_ID, compute_source_tree_hash
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json"
SPLIT_PATH = ROOT / "data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json"
VALIDATION_SET_PATH = ROOT / "data/manifests/PPO_PHASE3_VALIDATION_SET_V1.json"
PROTOCOL_PATH = ROOT / "data/manifests/PHASE3_VALIDATION_PROTOCOL_V1.json"
ARTIFACT_PATH = ROOT / "data/development/phase3_validation_common_model_v1.json"

SELECTION_POLICY_ID = "PPO_PHASE3_VALIDATION_SELECTION_V1"
PROTOCOL_ID = "PHASE3_VALIDATION_PROTOCOL_V1"
SEED_SET_ID = "PHASE3_VALIDATION_SEEDS_V1"
SEEDS = (20260928, 20260929, 20260930, 20261001, 20261002)
CHECKPOINTS = (5.0, 30.0, 60.0)
TIME_LIMIT = 60.0
METHOD_ALNS = "SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_OFF"
METHOD_HGA = "ADAPTED_HGA_V1"
METHOD_WAG = "ADAPTED_WAG_VNS_V1"
METHODS = (METHOD_ALNS, METHOD_HGA, METHOD_WAG)

TIERS = (
    ("SMALL", 20, 30, 25),
    ("MEDIUM", 50, 60, 55),
    ("LARGE", 80, 90, 85),
)
DESCRIPTOR_SOURCE_FIELDS = (
    "actual_weld_count",
    "total_weld_length_m",
    "bbox_area_ratio",
    "x_coverage_ratio",
    "y_coverage_ratio",
    "cross_y6_count",
    "upper_lower_length_imbalance",
    "left_right_length_imbalance",
)
DESCRIPTOR_OUTPUT_FIELDS = (
    "normalized_N",
    "total_weld_length",
    "bbox_coverage",
    "x_coverage",
    "y_coverage",
    "cross_y6_count",
    "upper_lower_imbalance",
    "left_right_imbalance",
)
CAP_HIT_FIELDS = (
    "vnd_candidate_cap_hits",
    "vnd_pass_cap_hits",
    "initialization_reference_limit_hits",
    "population_survival_events",
    "factorial_window_cap_hits",
    "factorial_call_cap_hits",
    "wag_variant_cap_hits",
    "route_combination_cap_hits",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _hash_payload(payload: Mapping[str, Any], hash_field: str) -> str:
    body = {key: value for key, value in payload.items() if key != hash_field}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def validation_alns_config() -> SearchConfig:
    return SearchConfig(
        construction_budget=5,
        kinit_ref=5,
        max_iterations=100000,
        time_limit=TIME_LIMIT,
        checkpoints=CHECKPOINTS,
        enable_two_opt_star=False,
    )


def method_configs() -> dict[str, Any]:
    return {
        METHOD_ALNS: validation_alns_config(),
        METHOD_HGA: AdaptedHGAConfig(),
        METHOD_WAG: AdaptedWAGConfig(),
    }


def method_config_hashes() -> dict[str, str]:
    return {
        method_id: canonical_config_hash(config)
        for method_id, config in method_configs().items()
    }


def _role_by_path(split: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item["relative_path"]): str(item["assigned_role"])
        for item in split["workbooks"]
    }


def _candidate_pool(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    lower: int,
    upper: int,
    target: int,
    unavailable_workbooks: set[str],
) -> list[Mapping[str, Any]]:
    roles = _role_by_path(split)
    eligible = [
        entry
        for entry in dataset["instances"]
        if entry.get("validation_status") == "VALID"
        and entry.get("duplicate_of") is None
        and 10 <= int(entry["actual_weld_count"]) <= 90
        and roles.get(str(entry["relative_path"])) == ROLE_VALIDATION
        and str(entry["relative_path"]) not in unavailable_workbooks
    ]
    preferred = [
        entry
        for entry in eligible
        if lower <= int(entry["actual_weld_count"]) <= upper
    ]
    if len({str(entry["relative_path"]) for entry in preferred}) >= 4:
        return preferred

    ordered_distances = sorted(
        {abs(int(entry["actual_weld_count"]) - target) for entry in eligible}
    )
    for distance in ordered_distances:
        expanded = [
            entry
            for entry in eligible
            if abs(int(entry["actual_weld_count"]) - target) <= distance
        ]
        if len({str(entry["relative_path"]) for entry in expanded}) >= 4:
            return expanded
    raise ValueError(f"fewer than four distinct VALIDATION workbooks for target N={target}")


def _raw_descriptor(entry: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(entry[field]) for field in DESCRIPTOR_SOURCE_FIELDS)


def _normalize_descriptors(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, ...]]:
    columns = list(zip(*(_raw_descriptor(entry) for entry in candidates)))
    minima = [min(column) for column in columns]
    maxima = [max(column) for column in columns]
    result: dict[str, tuple[float, ...]] = {}
    for entry in candidates:
        values = _raw_descriptor(entry)
        result[str(entry["instance_id"])] = tuple(
            0.0 if high == low else (value - low) / (high - low)
            for value, low, high in zip(values, minima, maxima)
        )
    return result


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _select_tier(
    candidates: Sequence[Mapping[str, Any]], target: int
) -> list[tuple[Mapping[str, Any], tuple[float, ...]]]:
    normalized = _normalize_descriptors(candidates)
    first = min(
        candidates,
        key=lambda entry: (
            abs(int(entry["actual_weld_count"]) - target),
            str(entry["instance_geometry_hash"]),
        ),
    )
    selected = [first]
    used = {str(first["relative_path"])}
    while len(selected) < 4:
        available = [
            entry
            for entry in candidates
            if str(entry["relative_path"]) not in used
        ]
        if not available:
            raise ValueError("geometry diversity selection exhausted distinct workbooks")

        def key(entry: Mapping[str, Any]) -> tuple[float, int, str]:
            vector = normalized[str(entry["instance_id"])]
            minimum_distance = min(
                _euclidean(vector, normalized[str(chosen["instance_id"])])
                for chosen in selected
            )
            return (
                -minimum_distance,
                abs(int(entry["actual_weld_count"]) - target),
                str(entry["instance_geometry_hash"]),
            )

        chosen = min(available, key=key)
        selected.append(chosen)
        used.add(str(chosen["relative_path"]))
    return [(entry, normalized[str(entry["instance_id"])]) for entry in selected]


def build_validation_set(
    dataset: Mapping[str, Any], split: Mapping[str, Any]
) -> dict[str, Any]:
    validate_phase3_split_manifest(split)
    if dataset["dataset_manifest_hash"] != split["dataset_manifest_hash"]:
        raise ValueError("dataset/split hash mismatch")
    selected: list[dict[str, Any]] = []
    used_workbooks: set[str] = set()
    ordinal = 0
    for tier, lower, upper, target in TIERS:
        pool = _candidate_pool(
            dataset,
            split,
            lower=lower,
            upper=upper,
            target=target,
            unavailable_workbooks=used_workbooks,
        )
        for entry, normalized in _select_tier(pool, target):
            ordinal += 1
            path = str(entry["relative_path"])
            used_workbooks.add(path)
            raw = _raw_descriptor(entry)
            selected.append(
                {
                    "selection_ordinal": ordinal,
                    "tier": tier,
                    "target_weld_count": target,
                    "workbook_path": path,
                    "relative_path": path,
                    "sheet_name": str(entry["sheet_name"]),
                    "instance_id": str(entry["instance_id"]),
                    "N": int(entry["actual_weld_count"]),
                    "actual_weld_count": int(entry["actual_weld_count"]),
                    "instance_geometry_hash": str(entry["instance_geometry_hash"]),
                    "phase3_role": ROLE_VALIDATION,
                    "descriptor": dict(zip(DESCRIPTOR_OUTPUT_FIELDS, raw)),
                    "normalized_descriptor": dict(
                        zip(DESCRIPTOR_OUTPUT_FIELDS, normalized)
                    ),
                }
            )
    payload: dict[str, Any] = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "selection_policy": {
            "main_range": [10, 90],
            "tiers": [
                {"tier": tier, "preferred_N": [low, high], "target_N": target}
                for tier, low, high, target in TIERS
            ],
            "descriptor_fields": list(DESCRIPTOR_OUTPUT_FIELDS),
            "normalization": "per-tier candidate-pool min-max",
            "first": "minimum |N-target|, tie instance_geometry_hash",
            "subsequent": (
                "farthest-first maximum minimum Euclidean descriptor distance; "
                "ties by |N-target| then instance_geometry_hash"
            ),
            "workbook_limit": 1,
            "solver_metrics_forbidden": True,
        },
        "instances": selected,
    }
    payload["validation_set_hash"] = _hash_payload(payload, "validation_set_hash")
    validate_validation_set(payload, split)
    return payload


def validate_validation_set(
    payload: Mapping[str, Any], split: Mapping[str, Any]
) -> None:
    if payload.get("selection_policy_id") != SELECTION_POLICY_ID:
        raise ValueError("unexpected validation selection policy")
    if payload.get("phase3_split_hash") != split.get("phase3_split_hash"):
        raise ValueError("validation set/split hash mismatch")
    if payload.get("validation_set_hash") != _hash_payload(
        payload, "validation_set_hash"
    ):
        raise ValueError("validation_set_hash mismatch")
    instances = list(payload.get("instances", ()))
    if len(instances) != 12:
        raise ValueError("validation set must contain exactly 12 instances")
    paths = [str(item["relative_path"]) for item in instances]
    if len(paths) != len(set(paths)):
        raise ValueError("validation set must use 12 unique workbooks")
    assert_solver_access_allowed(split, paths, allowed_roles=(ROLE_VALIDATION,))
    counts = {
        tier: sum(item["tier"] == tier for item in instances)
        for tier, *_ in TIERS
    }
    if counts != {"SMALL": 4, "MEDIUM": 4, "LARGE": 4}:
        raise ValueError(f"validation tiers must be 4/4/4, got {counts}")
    if any(item.get("phase3_role") != ROLE_VALIDATION for item in instances):
        raise ValueError("validation instance role must be VALIDATION")


def build_protocol(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    validation_set: Mapping[str, Any],
) -> dict[str, Any]:
    configs = method_configs()
    hashes = method_config_hashes()
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "validation_set_hash": validation_set["validation_set_hash"],
        "methods": [
            {
                "method_id": method_id,
                "method_config": asdict(configs[method_id]),
                "method_config_hash": hashes[method_id],
            }
            for method_id in METHODS
        ],
        "seed_set_id": SEED_SET_ID,
        "seeds": list(SEEDS),
        "time_limit": TIME_LIMIT,
        "checkpoints": list(CHECKPOINTS),
        "trajectory_policy": "one 60-second native trajectory per method-instance-seed",
        "initialization_timing": "included in wall clock",
        "overshoot_policy": (
            "a started formal reference evaluation may finish; post-deadline improvements "
            "must not backfill checkpoints"
        ),
        "selection_policy": validation_set["selection_policy"],
        "aggregation_policy": {
            "instance_method_seed_count": 5,
            "median_Cmax_60_requires": "5/5 certified checkpoint values",
            "backbone_ratio": "A_i / min(H_i, W_i)",
        },
        "release_rule": {
            "complete_certified_runs": 180,
            "numeric_failure_count": 0,
            "certifier_mismatch_count": 0,
            "median_ratio_max": 1.10,
            "instances_with_ratio_at_most_1_10_min": 8,
            "tier_median_ratio_max": 1.15,
            "iteration_limit_forbidden": True,
        },
    }
    payload["validation_protocol_hash"] = _hash_payload(
        payload, "validation_protocol_hash"
    )
    return payload


def validate_protocol(
    payload: Mapping[str, Any], validation_set: Mapping[str, Any]
) -> None:
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol id")
    if payload.get("validation_set_hash") != validation_set.get(
        "validation_set_hash"
    ):
        raise ValueError("protocol/validation-set hash mismatch")
    if payload.get("validation_protocol_hash") != _hash_payload(
        payload, "validation_protocol_hash"
    ):
        raise ValueError("validation_protocol_hash mismatch")
    if tuple(payload.get("seeds", ())) != SEEDS:
        raise ValueError("validation seed set changed")
    if tuple(payload.get("checkpoints", ())) != CHECKPOINTS:
        raise ValueError("validation checkpoints changed")
    protocol_hashes = {
        item["method_id"]: item["method_config_hash"] for item in payload["methods"]
    }
    if protocol_hashes != method_config_hashes():
        raise ValueError("validation method config changed")


def prepare_manifests(*, write: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _read_json(DATASET_PATH)
    split = _read_json(SPLIT_PATH)
    generated_set = build_validation_set(dataset, split)
    if VALIDATION_SET_PATH.exists():
        existing_set = _read_json(VALIDATION_SET_PATH)
        if _canonical_json(existing_set) != _canonical_json(generated_set):
            raise ValueError("frozen validation-set manifest differs from deterministic rebuild")
        validation_set = existing_set
    else:
        validation_set = generated_set
        if write:
            _write_json(VALIDATION_SET_PATH, validation_set)
    generated_protocol = build_protocol(dataset, split, validation_set)
    if PROTOCOL_PATH.exists():
        existing_protocol = _read_json(PROTOCOL_PATH)
        if _canonical_json(existing_protocol) != _canonical_json(generated_protocol):
            raise ValueError("frozen validation protocol differs from deterministic rebuild")
        protocol = existing_protocol
    else:
        protocol = generated_protocol
        if write:
            _write_json(PROTOCOL_PATH, protocol)
    validate_validation_set(validation_set, split)
    validate_protocol(protocol, validation_set)
    return validation_set, protocol


def _provenance(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    validation_set: Mapping[str, Any],
    protocol: Mapping[str, Any],
    revision_label: str,
) -> dict[str, Any]:
    return {
        "repository_id": REPOSITORY_ID,
        "explicit_local_revision_label": revision_label,
        "source_commit": revision_label,
        "source_tree_hash": compute_source_tree_hash(ROOT),
        "development_only": True,
        "commit_verified": False,
        "scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "validation_set_hash": validation_set["validation_set_hash"],
        "validation_protocol_hash": protocol["validation_protocol_hash"],
    }


def _run_key(
    provenance: Mapping[str, Any], method_id: str, instance_id: str, seed: int
) -> dict[str, Any]:
    return {
        "validation_set_hash": provenance["validation_set_hash"],
        "method_id": method_id,
        "instance_id": instance_id,
        "solver_seed": seed,
        "method_config_hash": method_config_hashes()[method_id],
        "scope_hash": provenance["scope_hash"],
        "source_tree_hash": provenance["source_tree_hash"],
    }


def _run_key_hash(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(key).encode("utf-8")).hexdigest()


def resume_record_matches(record: Mapping[str, Any], expected_key: Mapping[str, Any]) -> bool:
    return (
        record.get("run_key") == dict(expected_key)
        and record.get("run_key_hash") == _run_key_hash(expected_key)
    )


def load_validation_parents(
    split: Mapping[str, Any],
    entry: Mapping[str, Any],
    ppo_root: Path,
    *,
    loader=load_ppo_platform_instance,
) -> Sequence[Any]:
    """Authorize the frozen role before any workbook-facing function is called."""
    assert_solver_access_allowed(
        split,
        (str(entry["relative_path"]),),
        allowed_roles=(ROLE_VALIDATION,),
    )
    instance = loader(
        ppo_root / str(entry["relative_path"]),
        str(entry["sheet_name"]),
        ppo_root=ppo_root,
        instance_id=str(entry["instance_id"]),
    )
    return to_parent_welds(instance)


def _entry_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": entry["instance_id"],
        "workbook": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "N": entry["N"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
    }


def _baseline_record(
    result: Any,
    *,
    entry: Mapping[str, Any],
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    initial = result.initial_candidate
    telemetry: dict[str, Any] = {}
    if initial is not None and initial.directions is not None:
        telemetry = solution_telemetry(
            initial.solution, initial.directions, ScientificConfig()
        )
    accounting = result.accounting
    record = {
        **provenance,
        **_entry_fields(entry),
        **telemetry,
        "method_id": result.method_id,
        "method_config_hash": result.method_config_hash,
        "solver_seed": seed,
        "termination_reason": result.termination_reason,
        "initialization_status": (
            "SUCCESS" if initial is not None else result.status.value
        ),
        "initial_solution_hash": None if initial is None else initial.solution.canonical_hash,
        "initial_source": None if initial is None else initial.source,
        "initial_cmax": None if initial is None else initial.metrics.cmax,
        "time_to_first_certified": result.time_to_first_certified,
        "cmax_at_5": result.checkpoints.get(5.0),
        "cmax_at_30": result.checkpoints.get(30.0),
        "cmax_at_60": result.checkpoints.get(60.0),
        "final_cmax": None if result.metrics is None else result.metrics.cmax,
        "best_source": result.best_source,
        "iterations": result.iterations,
        "candidate_count": result.candidate_count,
        "reference_calls": result.reference_calls,
        "certifier_calls": result.certifier_calls,
        "direction_dp_calls": int(accounting["direction_dp_calls"]),
        "DEADLOCK": int(accounting["baseline_deadlock"]),
        "recovered": int(accounting["recovered"]),
        "remaining_DEADLOCK": int(accounting["remaining_deadlock"]),
        "scheduler_time": float(accounting["scheduler_time"]),
        "direction_time": float(accounting["direction_time"]),
        "local_search_time": float(accounting["local_search_time"]),
        "repair_time": 0.0,
        "construction_time": float(accounting["construction_time"]),
        "runtime": result.actual_runtime,
        "overshoot": result.overshoot,
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
        "cap_hit_telemetry": {
            field: int(accounting[field]) for field in CAP_HIT_FIELDS
        },
        "diagnostics": list(result.diagnostics),
    }
    return record


def _alns_record(
    result: Any,
    *,
    entry: Mapping[str, Any],
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    initialization = result.initialization
    telemetry: dict[str, Any] = {}
    if initialization.solution is not None and initialization.directions is not None:
        telemetry = solution_telemetry(
            initialization.solution, initialization.directions, ScientificConfig()
        )
    references = result.stats.reference_records
    record = {
        **provenance,
        **_entry_fields(entry),
        **telemetry,
        "method_id": METHOD_ALNS,
        "method_config_hash": canonical_config_hash(validation_alns_config()),
        "solver_seed": seed,
        "termination_reason": result.termination_reason,
        "initialization_status": initialization.status.value,
        "initial_solution_hash": (
            None if initialization.solution is None else initialization.solution.canonical_hash
        ),
        "initial_source": initialization.winning_strategy,
        "initial_cmax": (
            None if initialization.schedule is None else initialization.schedule.cmax
        ),
        "time_to_first_certified": (
            None if not result.stats.best_events else result.stats.best_events[0][0]
        ),
        "cmax_at_5": result.anytime.get(5.0, {}).get("cmax"),
        "cmax_at_30": result.anytime.get(30.0, {}).get("cmax"),
        "cmax_at_60": result.anytime.get(60.0, {}).get("cmax"),
        "final_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "best_source": (
            None
            if not result.stats.improvements_by_family
            else max(
                result.stats.improvements_by_family,
                key=result.stats.improvements_by_family.get,
            )
        ),
        "iterations": result.stats.iterations,
        "candidate_count": result.stats.constructed,
        "reference_calls": len(references),
        "certifier_calls": (
            sum(item["status"] == "FEASIBLE" for item in references) + 1
        ),
        "direction_dp_calls": (
            result.stats.direction_refinement_calls
            + sum(len(route) > 0 for route in (initialization.directions or ()))
        ),
        "DEADLOCK": sum(bool(item["baseline_deadlock"]) for item in references),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in references
        ),
        "remaining_DEADLOCK": sum(item["status"] == "DEADLOCK" for item in references),
        "scheduler_time": result.stats.reference_scheduler_time,
        "direction_time": result.stats.direction_dp_time,
        "local_search_time": result.stats.repair_time,
        "repair_time": result.stats.repair_time,
        "construction_time": result.stats.candidate_generation_time,
        "runtime": result.runtime,
        "overshoot": result.stats.overshoot,
        "final_schedule_status": (
            None if result.best_schedule is None else result.best_schedule.status.value
        ),
        "final_certified": bool(
            result.final_certification and result.final_certification.certified
        ),
        "numeric_failure_count": int(result.stats.n_numeric_failure)
        + int(result.status.value == "NUMERIC_FAILURE"),
        "certifier_mismatch_count": int(
            result.best_schedule is not None
            and result.best_schedule.status is ScheduleStatus.FEASIBLE
            and not (
                result.final_certification and result.final_certification.certified
            )
        ),
        "cap_hit_telemetry": {field: 0 for field in CAP_HIT_FIELDS},
        "move_telemetry": {
            move: {
                "attempted": result.stats.attempted_by_move[move],
                "constructed": result.stats.constructed_by_move[move],
                "c3": result.stats.c3_by_move[move],
                "c4": result.stats.c4_by_move[move],
                "accepted": result.stats.accepted_by_move[move],
                "global_best_improvement": result.stats.best_improvement_by_move[move],
            }
            for move in result.stats.attempted_by_move
        },
        "diagnostics": [],
    }
    return record


def _run_method(
    method_id: str,
    parents: Sequence[Any],
    *,
    entry: Mapping[str, Any],
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if method_id == METHOD_ALNS:
        result = run_bounded_sa_oi(
            parents,
            ScientificConfig(),
            validation_alns_config(),
            seed=seed,
            scope=FORMAL_SCOPE_V1_1,
            source_commit=str(provenance["explicit_local_revision_label"]),
            allow_unverified_source=True,
            formal_result=False,
        )
        record = _alns_record(result, entry=entry, seed=seed, provenance=provenance)
    elif method_id == METHOD_HGA:
        result = run_adapted_hga(
            parents,
            ScientificConfig(),
            AdaptedHGAConfig(),
            seed=seed,
            time_limit=TIME_LIMIT,
            checkpoints=CHECKPOINTS,
            scope=FORMAL_SCOPE_V1_1,
        )
        record = _baseline_record(result, entry=entry, seed=seed, provenance=provenance)
    elif method_id == METHOD_WAG:
        result = run_adapted_wag_vns(
            parents,
            ScientificConfig(),
            AdaptedWAGConfig(),
            seed=seed,
            time_limit=TIME_LIMIT,
            checkpoints=CHECKPOINTS,
            scope=FORMAL_SCOPE_V1_1,
        )
        record = _baseline_record(result, entry=entry, seed=seed, provenance=provenance)
    else:
        raise ValueError(f"unknown validation method: {method_id}")
    key = _run_key(provenance, method_id, str(entry["instance_id"]), seed)
    if record["method_config_hash"] != key["method_config_hash"]:
        raise RuntimeError("executed method config hash differs from frozen run key")
    record["run_key"] = key
    record["run_key_hash"] = _run_key_hash(key)
    return record


def checkpoint_median(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    if len(rows) != 5 or any(value is None for value in values):
        return None
    return float(statistics.median(float(value) for value in values))


def _iqr(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def build_summaries(
    records: Sequence[Mapping[str, Any]], validation_set: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    instance_summaries: list[dict[str, Any]] = []
    for entry in validation_set["instances"]:
        for method_id in METHODS:
            rows = [
                row
                for row in records
                if row["instance_id"] == entry["instance_id"]
                and row["method_id"] == method_id
            ]
            first = [
                float(row["time_to_first_certified"])
                for row in rows
                if row.get("time_to_first_certified") is not None
            ]
            c60 = [
                float(row["cmax_at_60"])
                for row in rows
                if row.get("cmax_at_60") is not None
            ]
            instance_summaries.append(
                {
                    "instance_id": entry["instance_id"],
                    "tier": entry["tier"],
                    "N": entry["N"],
                    "method_id": method_id,
                    "run_count": len(rows),
                    "certified_seed_count": sum(
                        bool(row.get("final_certified"))
                        and row.get("cmax_at_60") is not None
                        for row in rows
                    ),
                    "certified_at_5_count": sum(
                        row.get("cmax_at_5") is not None for row in rows
                    ),
                    "certified_at_30_count": sum(
                        row.get("cmax_at_30") is not None for row in rows
                    ),
                    "median_Cmax_5": checkpoint_median(rows, "cmax_at_5"),
                    "median_Cmax_30": checkpoint_median(rows, "cmax_at_30"),
                    "median_Cmax_60": checkpoint_median(rows, "cmax_at_60"),
                    "mean_Cmax_60": statistics.fmean(c60) if len(c60) == 5 else None,
                    "std_Cmax_60": statistics.stdev(c60) if len(c60) == 5 else None,
                    "IQR_Cmax_60": _iqr(c60) if len(c60) == 5 else None,
                    "median_time_to_first": (
                        float(statistics.median(first)) if len(first) == 5 else None
                    ),
                    "median_reference_calls": (
                        float(statistics.median(row["reference_calls"] for row in rows))
                        if len(rows) == 5
                        else None
                    ),
                    "median_runtime": (
                        float(statistics.median(row["runtime"] for row in rows))
                        if len(rows) == 5
                        else None
                    ),
                }
            )

    ratios: list[dict[str, Any]] = []
    for entry in validation_set["instances"]:
        by_method = {
            row["method_id"]: row
            for row in instance_summaries
            if row["instance_id"] == entry["instance_id"]
        }
        values = {
            method: by_method[method]["median_Cmax_60"] for method in METHODS
        }
        ratio = (
            None
            if any(value is None for value in values.values())
            else values[METHOD_ALNS]
            / min(values[METHOD_HGA], values[METHOD_WAG])
        )
        ratios.append(
            {
                "instance_id": entry["instance_id"],
                "tier": entry["tier"],
                "A_i": values[METHOD_ALNS],
                "H_i": values[METHOD_HGA],
                "W_i": values[METHOD_WAG],
                "R_i": ratio,
            }
        )

    tier_summaries = []
    for tier, *_ in TIERS:
        tier_ratios = [row["R_i"] for row in ratios if row["tier"] == tier]
        tier_summaries.append(
            {
                "tier": tier,
                "ratio_count": sum(value is not None for value in tier_ratios),
                "median_R_i": (
                    None
                    if any(value is None for value in tier_ratios)
                    else float(statistics.median(tier_ratios))
                ),
            }
        )
    return instance_summaries, tier_summaries, {"ratios": ratios}


def evaluate_release(
    records: Sequence[Mapping[str, Any]],
    tier_summaries: Sequence[Mapping[str, Any]],
    ratio_payload: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = [row["R_i"] for row in ratio_payload["ratios"]]
    complete = len(records) == 180
    certified = complete and all(
        bool(row.get("final_certified")) and row.get("cmax_at_60") is not None
        for row in records
    )
    numeric_failures = sum(int(row.get("numeric_failure_count", 0)) for row in records)
    mismatches = sum(int(row.get("certifier_mismatch_count", 0)) for row in records)
    iteration_limits = sum(
        row.get("termination_reason") == "ITERATION_LIMIT" for row in records
    )
    ratio_complete = len(ratios) == 12 and all(value is not None for value in ratios)
    median_ratio = (
        float(statistics.median(ratios)) if ratio_complete else None
    )
    within = (
        sum(value <= 1.10 for value in ratios) if ratio_complete else None
    )
    tier_pass = len(tier_summaries) == 3 and all(
        row["median_R_i"] is not None and row["median_R_i"] <= 1.15
        for row in tier_summaries
    )
    experiment_valid = (
        complete
        and certified
        and numeric_failures == 0
        and mismatches == 0
        and iteration_limits == 0
    )
    backbone_pass = (
        experiment_valid
        and ratio_complete
        and median_ratio <= 1.10
        and within >= 8
        and tier_pass
    )
    return {
        "PHASE3_VALIDATION_STATUS": "PASS" if experiment_valid else "FAIL",
        "DETERMINISTIC_BACKBONE_STATUS": (
            "FROZEN" if backbone_pass else "NEEDS_CANDIDATE_POOL_AUDIT"
        ),
        "FROZEN_PROPOSED_BACKBONE": (
            METHOD_ALNS if backbone_pass else None
        ),
        "TWO_OPT_STAR_STATUS": "ABLATION_ONLY",
        "ID_TEST_STATUS": "SEALED",
        "NEXT_PHASE": "Phase 4-0 — Candidate-Pool Oracle Recall Audit",
        "checks": {
            "run_count": len(records),
            "complete_180": complete,
            "certified_180": certified,
            "numeric_failure_count": numeric_failures,
            "certifier_mismatch_count": mismatches,
            "iteration_limit_count": iteration_limits,
            "ratio_count": sum(value is not None for value in ratios),
            "median_R_i": median_ratio,
            "instances_R_i_at_most_1_10": within,
            "tier_medians_at_most_1_15": tier_pass,
            "backbone_gate_pass": backbone_pass,
        },
    }


def _refresh_artifact(
    artifact: dict[str, Any], validation_set: Mapping[str, Any]
) -> None:
    records = artifact["records"]
    instance, tiers, ratios = build_summaries(records, validation_set)
    artifact["per_instance_summaries"] = instance
    artifact["tier_summaries"] = tiers
    artifact["backbone_ratios"] = ratios["ratios"]
    artifact["release_decision"] = evaluate_release(records, tiers, ratios)


def _new_artifact(
    protocol: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_id": "PHASE3_VALIDATION_COMMON_MODEL_V1",
        "evidence_role": "VALIDATION_TUNING_EVIDENCE_NOT_ID_TEST_FORMAL_RESULT",
        "protocol": dict(protocol),
        "selection_manifest_hash": provenance["validation_set_hash"],
        "method_configs": protocol["methods"],
        "seeds": list(SEEDS),
        "records": [],
        "per_instance_summaries": [],
        "tier_summaries": [],
        "backbone_ratios": [],
        "release_decision": {},
        "provenance": dict(provenance),
    }


def _load_or_create_artifact(
    protocol: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        return _new_artifact(protocol, provenance)
    artifact = _read_json(ARTIFACT_PATH)
    if artifact.get("protocol", {}).get("validation_protocol_hash") != protocol.get(
        "validation_protocol_hash"
    ):
        raise ValueError("resume rejected: validation protocol mismatch")
    if artifact.get("provenance") != dict(provenance):
        raise ValueError("resume rejected: provenance mismatch")
    return artifact


def run_validation(ppo_root: Path, revision_label: str) -> dict[str, Any]:
    dataset = _read_json(DATASET_PATH)
    split = _read_json(SPLIT_PATH)
    validation_set, protocol = prepare_manifests(write=True)
    provenance = _provenance(
        dataset, split, validation_set, protocol, revision_label
    )
    artifact = _load_or_create_artifact(protocol, provenance)
    existing_by_hash = {
        str(record["run_key_hash"]): record for record in artifact["records"]
    }
    expected_hashes = set()
    for entry in validation_set["instances"]:
        for method_id in METHODS:
            for seed in SEEDS:
                key = _run_key(
                    provenance, method_id, str(entry["instance_id"]), seed
                )
                key_hash = _run_key_hash(key)
                expected_hashes.add(key_hash)
                if key_hash in existing_by_hash:
                    if not resume_record_matches(existing_by_hash[key_hash], key):
                        raise ValueError("resume rejected: exact-key mismatch")
                    continue
                parents = load_validation_parents(split, entry, ppo_root)
                record = _run_method(
                    method_id,
                    parents,
                    entry=entry,
                    seed=seed,
                    provenance=provenance,
                )
                if record["run_key_hash"] != key_hash:
                    raise RuntimeError("completed run identity differs from expected run key")
                artifact["records"].append(record)
                existing_by_hash[key_hash] = record
                _refresh_artifact(artifact, validation_set)
                _write_json(ARTIFACT_PATH, artifact)
                print(
                    f"completed {len(artifact['records'])}/180 "
                    f"{method_id} {entry['instance_id']} seed={seed}",
                    flush=True,
                )
                if (
                    record["certifier_mismatch_count"]
                    or record["numeric_failure_count"]
                    or record["termination_reason"] == "ITERATION_LIMIT"
                ):
                    raise RuntimeError(
                        "VALIDATION protocol failure; batch stopped after persisting run"
                    )
    unexpected = set(existing_by_hash) - expected_hashes
    if unexpected:
        raise ValueError("resume rejected: artifact contains unexpected run keys")
    _refresh_artifact(artifact, validation_set)
    _write_json(ARTIFACT_PATH, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen Phase 3-2B VALIDATION runner")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument(
        "--source-commit-label",
        default=(
            "KNOWN_MAIN_8321c46e4640f4f6928cee3f951b3e9f8b848f95"
            "+LOCAL_UNCOMMITTED_DRL_TREE"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    validation_set, protocol = prepare_manifests(write=True)
    print(
        f"validation_set_hash={validation_set['validation_set_hash']}\n"
        f"validation_protocol_hash={protocol['validation_protocol_hash']}",
        flush=True,
    )
    if args.prepare_only:
        return 0
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    artifact = run_validation(Path(args.ppo_root).resolve(), args.source_commit_label)
    decision = artifact["release_decision"]
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    return 0 if decision["PHASE3_VALIDATION_STATUS"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
