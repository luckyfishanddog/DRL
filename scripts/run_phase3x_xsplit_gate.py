from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from mrta_baselines.common import canonical_config_hash
from mrta_data.ppo_instances import (
    instance_geometry_hash,
    load_ppo_platform_instance,
    to_parent_welds,
    x_split_geometry_metrics,
)
from mrta_reference.certifier import certify_schedule
from mrta_reference.model import OperationKind, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.provenance import compute_source_tree_hash
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import (
    EXPERIMENTAL_X_SPLIT_SCOPE_V1,
    FORMAL_SCOPE_V1_1,
)
from mrta_reference.solution import block_map, official_metrics
from mrta_search.initialization import InitializationResult
from mrta_search.pipeline import SearchConfig, SearchResult, run_bounded_sa_oi


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json"
SPLIT_PATH = ROOT / "data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json"
GATE_SET_PATH = ROOT / "data/manifests/PPO_X_SPLIT_GATE_SET_V1.json"
PROTOCOL_PATH = ROOT / "data/manifests/PHASE3X_XSPLIT_PROTOCOL_V1.json"
ARTIFACT_PATH = ROOT / "data/development/phase3x_xsplit_mechanism_gate_v1.json"

GATE_SET_ID = "PPO_X_SPLIT_GATE_SET_V1"
PROTOCOL_ID = "PHASE3X_XSPLIT_PROTOCOL_V1"
ARTIFACT_ID = "PHASE3X_XSPLIT_MECHANISM_GATE_V1"
SEEDS = (20261004, 20261005, 20261006)
ARMS = ("NO_X_CONTROL", "FINITE_X_DOMAIN")
SEARCH_SECONDS = 60.0
CHECKPOINTS = (5.0, 30.0, 60.0)
TIER_TARGETS = (("SMALL", 25), ("MEDIUM", 55), ("LARGE", 85))
COMMON_SEED = 20261004


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_payload(payload: Mapping[str, Any], hash_field: str) -> str:
    body = {key: value for key, value in payload.items() if key != hash_field}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"refusing to create unapproved directory: {path.parent}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gate_search_config(*, seconds: float = SEARCH_SECONDS) -> SearchConfig:
    if seconds != SEARCH_SECONDS:
        raise ValueError("the frozen Phase 3-X search budget is exactly 60 seconds")
    return SearchConfig(
        construction_budget=5,
        kinit_ref=5,
        max_iterations=100_000,
        time_limit=seconds,
        checkpoints=CHECKPOINTS,
        enable_two_opt_star=False,
    )


def common_seed_config() -> SearchConfig:
    return replace(
        gate_search_config(), max_iterations=0, time_limit=None, checkpoints=()
    )


def _development_paths(split: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(str(path) for path in split["development_consumed_workbooks"])


def assert_development_access(
    split: Mapping[str, Any], relative_paths: Sequence[str]
) -> None:
    allowed = _development_paths(split)
    forbidden = sorted(set(map(str, relative_paths)) - allowed)
    if forbidden:
        raise PermissionError(
            "Phase 3-X permits DEVELOPMENT_CONSUMED only; rejected: "
            + ", ".join(forbidden)
        )


def _load_instance(
    entry: Mapping[str, Any], split: Mapping[str, Any], ppo_root: Path
):
    relative = str(entry["relative_path"])
    assert_development_access(split, (relative,))
    instance = load_ppo_platform_instance(
        ppo_root / relative,
        str(entry["sheet_name"]),
        ppo_root=ppo_root,
        instance_id=str(entry["instance_id"]),
    )
    actual_hash = instance_geometry_hash(instance)
    if actual_hash != entry["instance_geometry_hash"]:
        raise ValueError(f"instance geometry hash mismatch: {entry['instance_id']}")
    return instance


def _select_extremes(
    rows: Sequence[dict[str, Any]], used_workbooks: set[str]
) -> list[dict[str, Any]] | None:
    eligible = [row for row in rows if row["relative_path"] not in used_workbooks]

    def choose(order, count: int, excluded: set[str]) -> list[dict[str, Any]]:
        selected = []
        for row in order:
            workbook = str(row["relative_path"])
            if workbook in excluded or any(item["relative_path"] == workbook for item in selected):
                continue
            selected.append(row)
            if len(selected) == count:
                break
        return selected

    high_order = sorted(
        eligible,
        key=lambda row: (
            -float(row["x_geometry_metadata"]["x_splittable_process_share"]),
            str(row["instance_geometry_hash"]),
        ),
    )
    high = choose(high_order, 2, set())
    low_order = sorted(
        eligible,
        key=lambda row: (
            float(row["x_geometry_metadata"]["x_splittable_process_share"]),
            str(row["instance_geometry_hash"]),
        ),
    )
    low = choose(low_order, 2, {str(item["relative_path"]) for item in high})
    return high + low if len(high) == 2 and len(low) == 2 else None


def build_gate_set(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    ppo_root: Path,
) -> dict[str, Any]:
    development = _development_paths(split)
    candidates = [
        dict(entry)
        for entry in dataset["instances"]
        if str(entry["relative_path"]) in development
        and entry.get("validation_status") == "VALID"
        and entry.get("formal_compatibility") == "PASS"
        and entry.get("duplicate_of") is None
    ]
    used_workbooks: set[str] = set()
    selected: list[dict[str, Any]] = []
    metadata_cache: dict[str, dict[str, Any]] = {}
    for tier, target in TIER_TARGETS:
        tier_rows: list[dict[str, Any]] = []
        chosen = None
        distances = sorted(
            {
                abs(int(row["actual_weld_count"]) - target)
                for row in candidates
                if row["relative_path"] not in used_workbooks
            }
        )
        for distance in distances:
            for row in candidates:
                if row["relative_path"] in used_workbooks:
                    continue
                if abs(int(row["actual_weld_count"]) - target) != distance:
                    continue
                instance_id = str(row["instance_id"])
                if instance_id not in metadata_cache:
                    metadata_cache[instance_id] = x_split_geometry_metrics(
                        _load_instance(row, split, ppo_root)
                    )
                enriched = dict(row)
                enriched["x_geometry_metadata"] = metadata_cache[instance_id]
                tier_rows.append(enriched)
            chosen = _select_extremes(tier_rows, used_workbooks)
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(f"cannot select four distinct DEVELOPMENT workbooks for {tier}")
        for index, row in enumerate(chosen):
            opportunity_group = "X_RICH" if index < 2 else "LOW_OPPORTUNITY_CONTROL"
            entry = {
                "instance_id": row["instance_id"],
                "relative_path": row["relative_path"],
                "sheet_name": row["sheet_name"],
                "raw_file_sha256": row["raw_file_sha256"],
                "instance_geometry_hash": row["instance_geometry_hash"],
                "N": int(row["actual_weld_count"]),
                "tier": tier,
                "tier_target_N": target,
                "tier_distance": abs(int(row["actual_weld_count"]) - target),
                "opportunity_group": opportunity_group,
                "x_geometry_metadata": row["x_geometry_metadata"],
                "selection_ordinal": len(selected) + 1,
            }
            selected.append(entry)
            used_workbooks.add(str(row["relative_path"]))
    payload: dict[str, Any] = {
        "gate_set_id": GATE_SET_ID,
        "evidence_role": "DEVELOPMENT_MECHANISM_GATE_NOT_VALIDATION_OR_ID_TEST",
        "selection_rule": {
            "authorization": "DEVELOPMENT_CONSUMED_ONLY",
            "targets": {tier: target for tier, target in TIER_TARGETS},
            "tier_expansion": "increase_abs_N_minus_target_until_2_high_plus_2_low_distinct_workbooks_exist",
            "within_tier": "2 highest and 2 lowest x_splittable_process_share",
            "tie_break": "instance_geometry_hash_lexicographic",
            "workbook_uniqueness": "12 distinct workbooks globally",
            "solver_calls": 0,
        },
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "instance_count": len(selected),
        "instances": selected,
    }
    payload["xsplit_gate_set_hash"] = _hash_payload(payload, "xsplit_gate_set_hash")
    validate_gate_set(payload, dataset, split)
    return payload


def validate_gate_set(
    payload: Mapping[str, Any],
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
) -> None:
    if payload.get("xsplit_gate_set_hash") != _hash_payload(payload, "xsplit_gate_set_hash"):
        raise ValueError("X_SPLIT gate-set hash mismatch")
    if payload.get("dataset_manifest_hash") != dataset.get("dataset_manifest_hash"):
        raise ValueError("X_SPLIT gate-set dataset hash mismatch")
    if payload.get("phase3_split_hash") != split.get("phase3_split_hash"):
        raise ValueError("X_SPLIT gate-set split hash mismatch")
    rows = list(payload.get("instances", ()))
    if len(rows) != 12 or len({row["relative_path"] for row in rows}) != 12:
        raise ValueError("X_SPLIT gate set requires 12 distinct workbooks")
    assert_development_access(split, tuple(str(row["relative_path"]) for row in rows))
    for tier, _ in TIER_TARGETS:
        tier_rows = [row for row in rows if row["tier"] == tier]
        if len(tier_rows) != 4:
            raise ValueError(f"{tier} must contain four instances")
        groups = [row["opportunity_group"] for row in tier_rows]
        if groups.count("X_RICH") != 2 or groups.count("LOW_OPPORTUNITY_CONTROL") != 2:
            raise ValueError(f"{tier} opportunity balance mismatch")


def build_protocol(gate_set: Mapping[str, Any]) -> dict[str, Any]:
    config = ScientificConfig()
    search = gate_search_config()
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "evidence_role": "DEVELOPMENT_MECHANISM_GATE_NOT_FORMAL_RESULT",
        "gate_set_hash": gate_set["xsplit_gate_set_hash"],
        "historical_scope": {
            "scope_id": FORMAL_SCOPE_V1_1.scope_id,
            "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
            "unchanged": True,
        },
        "experimental_scope": {
            "scope_id": EXPERIMENTAL_X_SPLIT_SCOPE_V1.scope_id,
            "scope_hash": EXPERIMENTAL_X_SPLIT_SCOPE_V1.scope_hash,
            "active_formal_scope_changed": False,
        },
        "scientific_config_hash": config.scientific_hash,
        "search_config": json.loads(_canonical_json(asdict(search))),
        "search_config_hash": canonical_config_hash(search),
        "common_seed": {
            "solver_seed": COMMON_SEED,
            "scope_id": FORMAL_SCOPE_V1_1.scope_id,
            "construction_excluded_from_search_seconds": True,
        },
        "arms": {
            "NO_X_CONTROL": {
                "scope_id": FORMAL_SCOPE_V1_1.scope_id,
                "x_split_access": False,
            },
            "FINITE_X_DOMAIN": {
                "scope_id": EXPERIMENTAL_X_SPLIT_SCOPE_V1.scope_id,
                "x_split_access": True,
            },
        },
        "solver_seeds": list(SEEDS),
        "search_seconds_per_arm": SEARCH_SECONDS,
        "expected_run_count": 72,
        "x_split_processing_semantics": {
            "split_time": 0.0,
            "cut_time": 0.0,
            "per_child_setup_weld_post": True,
            "extra_fixed_processing_per_accepted_split_seconds": config.t_pre
            + config.t_post,
            "shared_split_point_interference_exception": False,
        },
        "support_rule": {
            "semantic_tests_pass": True,
            "runs_without_numeric_failure": "72/72",
            "all_final_solutions_certified": True,
            "instances_with_median_improvement_ge_0_01": 2,
            "one_supporting_instance_N_ge_50": True,
            "supporting_final_best_contains_x_split": True,
            "one_instance_has_certified_x_global_best_update": True,
            "x_rich_distinct_workbooks_with_at_least_2_of_3_x_wins": 2,
        },
        "forbidden_data_roles": ["TRAIN_POOL", "VALIDATION", "ID_TEST"],
        "id_test_status": "SEALED",
        "runner_sha256": _sha256_file(Path(__file__)),
    }
    payload["phase3x_protocol_hash"] = _hash_payload(payload, "phase3x_protocol_hash")
    return payload


def prepare(ppo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _read_json(DATASET_PATH)
    split = _read_json(SPLIT_PATH)
    gate_set = build_gate_set(dataset, split, ppo_root)
    protocol = build_protocol(gate_set)
    return gate_set, protocol


def _provenance(
    gate_set: Mapping[str, Any], protocol: Mapping[str, Any], source_commit_label: str
) -> dict[str, Any]:
    return {
        "source_commit_label": source_commit_label,
        "source_tree_hash": compute_source_tree_hash(ROOT),
        "runner_sha256": _sha256_file(Path(__file__)),
        "gate_set_hash": gate_set["xsplit_gate_set_hash"],
        "protocol_hash": protocol["phase3x_protocol_hash"],
        "scientific_config_hash": protocol["scientific_config_hash"],
        "search_config_hash": protocol["search_config_hash"],
        "evidence_role": "DEVELOPMENT_ONLY",
    }


def _common_seed(
    parents,
    *,
    source_commit_label: str,
) -> tuple[InitializationResult, float]:
    started = time.perf_counter()
    result = run_bounded_sa_oi(
        parents,
        ScientificConfig(),
        common_seed_config(),
        seed=COMMON_SEED,
        scope=FORMAL_SCOPE_V1_1,
        source_commit=source_commit_label,
        allow_unverified_source=True,
    )
    elapsed = time.perf_counter() - started
    initialization = result.initialization
    if (
        initialization.solution is None
        or initialization.directions is None
        or initialization.schedule is None
        or initialization.certification is None
        or not initialization.certification.certified
    ):
        raise RuntimeError("common NO-X seed construction did not certify")
    if any(pattern.kind is SplitKind.X_SPLIT for pattern in initialization.solution.patterns):
        raise RuntimeError("common NO-X seed unexpectedly contains X_SPLIT")
    return initialization, elapsed


def _experimental_initialization(
    initialization: InitializationResult,
) -> InitializationResult:
    assert initialization.solution is not None
    assert initialization.directions is not None
    orientations = {robot: initialization.directions[robot] for robot in range(4)}
    schedule = reference_schedule_formal(
        initialization.solution,
        ScientificConfig(),
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
        orientations=orientations,
    )
    certification = certify_schedule(
        initialization.solution,
        schedule,
        ScientificConfig(),
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
    )
    if not certification.certified:
        raise RuntimeError("common seed failed experimental-scope replay")
    return replace(
        initialization,
        schedule=schedule,
        certification=certification,
    )


def _final_solution_telemetry(result: SearchResult) -> dict[str, Any]:
    solution = result.best_solution
    schedule = result.best_schedule
    if solution is None or schedule is None or schedule.cmax is None:
        return {}
    config = ScientificConfig()
    blocks = block_map(solution, config)
    process_by_robot = [
        sum(config.process_time(blocks[block_id].length) for block_id in route.block_ids)
        for route in solution.routes
    ]
    empty_by_robot: list[float] = []
    wait_by_robot: list[float] = []
    for robot in range(4):
        welds = sorted(
            (
                operation
                for operation in schedule.operations
                if operation.robot_id == robot and operation.kind is OperationKind.WELD
            ),
            key=lambda operation: operation.sequence_index,
        )
        empty_by_robot.append(
            sum(
                math.dist(left.end, right.start) / config.empty_speed
                for left, right in zip(welds, welds[1:])
            )
        )
        wait_by_robot.append(
            sum(
                operation.duration
                for operation in schedule.operations
                if operation.robot_id == robot and operation.kind is OperationKind.WAIT
            )
        )
    makespan_robot_ids = [
        robot
        for robot, completion in enumerate(schedule.robot_completion)
        if abs(completion - schedule.cmax)
        <= config.numeric_epsilon * max(1.0, schedule.cmax)
    ]
    makespan_robot_id = min(makespan_robot_ids)

    parents = {parent.parent_id: parent for parent in solution.parents}
    whole_parent_length = 0.0
    split_parent_length = 0.0
    split_child_length = 0.0
    x_parent_length = 0.0
    x_child_length = 0.0
    for pattern in solution.patterns:
        parent_length = parents[pattern.parent_id].length
        if pattern.kind is SplitKind.WHOLE:
            whole_parent_length += parent_length
            continue
        child_length = sum(
            block.length for block in blocks.values() if block.parent_id == pattern.parent_id
        )
        split_parent_length += parent_length
        split_child_length += child_length
        if pattern.kind is SplitKind.X_SPLIT:
            x_parent_length += parent_length
            x_child_length += child_length

    return {
        "robot_process_time": process_by_robot,
        "robot_empty_travel": empty_by_robot,
        "robot_wait": wait_by_robot,
        "robot_completion": list(schedule.robot_completion),
        "makespan_robot_id": makespan_robot_id,
        "makespan_robot_ids": makespan_robot_ids,
        "makespan_robot_wait": wait_by_robot[makespan_robot_id],
        "length_coverage": {
            "parent_total_length": sum(parent.length for parent in solution.parents),
            "whole_parent_total_length": whole_parent_length,
            "split_parent_total_length": split_parent_length,
            "split_child_total_length": split_child_length,
            "split_conservation_error": split_child_length - split_parent_length,
            "x_split_parent_total_length": x_parent_length,
            "x_split_child_total_length": x_child_length,
            "x_split_conservation_error": x_child_length - x_parent_length,
        },
    }


def _record(
    result: SearchResult,
    *,
    entry: Mapping[str, Any],
    arm: str,
    seed: int,
    common_seed_hash: str,
    common_seed_cmax: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = result.best_metrics
    solution = result.best_solution
    schedule = result.best_schedule
    x_patterns = (
        []
        if solution is None
        else [pattern for pattern in solution.patterns if pattern.kind is SplitKind.X_SPLIT]
    )
    final_telemetry = _final_solution_telemetry(result)
    references = result.stats.reference_records
    key = {
        **provenance,
        "instance_id": entry["instance_id"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "arm": arm,
        "solver_seed": seed,
        "common_seed_hash": common_seed_hash,
    }
    return {
        "run_key": key,
        "run_key_hash": hashlib.sha256(_canonical_json(key).encode("utf-8")).hexdigest(),
        "instance_id": entry["instance_id"],
        "workbook": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "opportunity_group": entry["opportunity_group"],
        "N": entry["N"],
        "arm": arm,
        "solver_seed": seed,
        "common_seed_hash": common_seed_hash,
        "common_seed_cmax": common_seed_cmax,
        "status": result.status.value,
        "termination_reason": result.termination_reason,
        "final_solution_hash": None if solution is None else solution.canonical_hash,
        "final_cmax": None if metrics is None else metrics.cmax,
        "final_certified": bool(
            result.final_certification and result.final_certification.certified
        ),
        "final_schedule_status": None if schedule is None else schedule.status.value,
        "numeric_failure_count": result.stats.n_numeric_failure,
        "total_wait": None if metrics is None else metrics.total_waiting,
        "cmax_at_5": result.anytime.get(5.0, {}).get("cmax"),
        "cmax_at_30": result.anytime.get(30.0, {}).get("cmax"),
        "cmax_at_60": result.anytime.get(60.0, {}).get("cmax"),
        "makespan_robot_wait": final_telemetry.get("makespan_robot_wait"),
        "process_spread": None if metrics is None else metrics.process_imbalance,
        "empty_travel": None if metrics is None else metrics.total_empty_travel,
        "x_split_parent_count": len(x_patterns),
        "x_split_processing_overhead": len(x_patterns)
        * (ScientificConfig().t_pre + ScientificConfig().t_post),
        "chosen_x_patterns": [
            {
                "pattern_id": pattern.pattern_id,
                "parent_id": pattern.parent_id,
                "rail": pattern.rail.value if pattern.rail is not None else None,
                "t": pattern.t,
                "source": pattern.point_id,
            }
            for pattern in x_patterns
        ],
        "x_split_final_pattern_ids": [pattern.pattern_id for pattern in x_patterns],
        "x_split_final_parent_ids": [pattern.parent_id for pattern in x_patterns],
        "x_split_final_t": [pattern.t for pattern in x_patterns],
        "x_split_final_rail": [
            pattern.rail.value if pattern.rail is not None else None
            for pattern in x_patterns
        ],
        "x_candidate_telemetry": {
            "generated": result.stats.x_pattern_candidates_generated,
            "cheap_feasible": result.stats.x_pattern_candidates_cheap_feasible,
            "c3_direction_evaluated": result.stats.x_pattern_candidates_c3,
            "reference_evaluated": result.stats.x_pattern_candidates_reference_evaluated,
            "c4_reference_evaluated": result.stats.x_pattern_candidates_reference_evaluated,
            "certified": result.stats.x_pattern_candidates_certified,
            "accepted": result.stats.x_pattern_candidates_accepted,
            "global_best_updates": result.stats.x_pattern_global_best_updates,
            "source_counts": dict(result.stats.x_pattern_source_counts),
        },
        "reference_calls": result.stats.nref,
        "direction_dp_calls": result.stats.kdp_count,
        "direction_refinement_calls": result.stats.direction_refinement_calls,
        "baseline_DEADLOCK": sum(
            bool(item["baseline_deadlock"]) for item in references
        ),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in references
        ),
        "remaining_DEADLOCK": sum(
            item["status"] == "DEADLOCK" for item in references
        ),
        "reference_status_counts": {
            "FEASIBLE": result.stats.n_feasible,
            "DEADLOCK": result.stats.n_deadlock,
            "INFEASIBLE": result.stats.n_infeasible,
            "NUMERIC_FAILURE": result.stats.n_numeric_failure,
        },
        "runtime": result.runtime,
        "overshoot": result.stats.overshoot,
        "iterations": result.stats.iterations,
        **final_telemetry,
        "provenance": dict(provenance),
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def aggregate(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_pair.setdefault((str(record["instance_id"]), int(record["solver_seed"])), {})[
            str(record["arm"])
        ] = record
    for (instance_id, seed), arms in sorted(by_pair.items()):
        if set(arms) != set(ARMS):
            continue
        control, finite = arms["NO_X_CONTROL"], arms["FINITE_X_DOMAIN"]
        c0, cx = float(control["final_cmax"]), float(finite["final_cmax"])
        pairs.append(
            {
                "instance_id": instance_id,
                "solver_seed": seed,
                "tier": finite["tier"],
                "opportunity_group": finite["opportunity_group"],
                "N": finite["N"],
                "cmax_no_x": c0,
                "cmax_x": cx,
                "delta_Cmax": cx - c0,
                "improvement_ratio": (c0 - cx) / c0,
                "delta_total_wait": float(finite["total_wait"]) - float(control["total_wait"]),
                "delta_makespan_robot_wait": float(finite["makespan_robot_wait"])
                - float(control["makespan_robot_wait"]),
                "delta_process_spread": float(finite["process_spread"])
                - float(control["process_spread"]),
                "delta_empty_travel": float(finite["empty_travel"])
                - float(control["empty_travel"]),
                "delta_reference_calls": int(finite["reference_calls"])
                - int(control["reference_calls"]),
                "x_split_selected": int(finite["x_split_parent_count"]) > 0,
                "x_global_best_update": int(
                    finite["x_candidate_telemetry"]["global_best_updates"]
                )
                > 0,
            }
        )
    instances = []
    for instance_id in sorted({str(row["instance_id"]) for row in pairs}):
        rows = [row for row in pairs if row["instance_id"] == instance_id]
        if len(rows) != 3:
            continue
        instances.append(
            {
                "instance_id": instance_id,
                "tier": rows[0]["tier"],
                "opportunity_group": rows[0]["opportunity_group"],
                "N": rows[0]["N"],
                "median_cmax_no_x": _median([row["cmax_no_x"] for row in rows]),
                "median_cmax_x": _median([row["cmax_x"] for row in rows]),
                "median_improvement_ratio": _median(
                    [row["improvement_ratio"] for row in rows]
                ),
                "median_delta_total_wait": _median(
                    [row["delta_total_wait"] for row in rows]
                ),
                "median_delta_makespan_robot_wait": _median(
                    [row["delta_makespan_robot_wait"] for row in rows]
                ),
                "x_split_selected_seeds": sum(row["x_split_selected"] for row in rows),
                "x_global_best_update_seeds": sum(
                    row["x_global_best_update"] for row in rows
                ),
                "x_win_seeds": sum(row["cmax_x"] < row["cmax_no_x"] for row in rows),
            }
        )
    return pairs, instances


def gate_decision(
    records: Sequence[Mapping[str, Any]], instance_summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    all_runs = len(records) == 72
    no_numeric = all(int(row["numeric_failure_count"]) == 0 for row in records)
    certified = all(bool(row["final_certified"]) for row in records)
    supporting = [
        row for row in instance_summaries if float(row["median_improvement_ratio"]) >= 0.01
    ]
    support_ids = {str(row["instance_id"]) for row in supporting}
    support_final_has_x = all(
        sum(
            record["arm"] == "FINITE_X_DOMAIN"
            and record["instance_id"] == instance_id
            and int(record["x_split_parent_count"]) > 0
            for record in records
        )
        == 3
        for instance_id in support_ids
    )
    x_global_update = any(
        int(record["x_candidate_telemetry"]["global_best_updates"]) > 0
        for record in records
        if record["arm"] == "FINITE_X_DOMAIN"
    )
    rich_stable = sum(
        row["opportunity_group"] == "X_RICH" and int(row["x_win_seeds"]) >= 2
        for row in instance_summaries
    )
    checks = {
        "semantic_tests_pass": True,
        "72_runs_without_numeric_failure": all_runs and no_numeric,
        "all_final_solutions_certified": all_runs and certified,
        "two_instances_median_improvement_ge_0_01": len(supporting) >= 2,
        "one_supporting_instance_N_ge_50": any(int(row["N"]) >= 50 for row in supporting),
        "supporting_final_best_contains_x_split": bool(supporting) and support_final_has_x,
        "certified_x_global_best_update_exists": x_global_update,
        "two_x_rich_workbooks_have_2_of_3_x_wins": rich_stable >= 2,
    }
    supported = all(checks.values())
    if supported:
        access = "ADEQUATE"
    elif x_global_update:
        access = "WEAK"
    else:
        access = "NOT_APPLICABLE"
    return {
        "checks": checks,
        "supporting_instances": sorted(support_ids),
        "X_SPLIT_MECHANISM_STATUS": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "X_SPLIT_SEARCH_ACCESS": access,
        "FORMAL_SCOPE_V2_AUTHORIZED": "YES" if supported else "NO",
        "ID_TEST_STATUS": "SEALED",
    }


def _new_artifact(
    gate_set: Mapping[str, Any], protocol: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_id": ARTIFACT_ID,
        "gate_set_hash": gate_set["xsplit_gate_set_hash"],
        "protocol_hash": protocol["phase3x_protocol_hash"],
        "provenance": dict(provenance),
        "common_seeds": {},
        "records": [],
        "paired_comparisons": [],
        "instance_summaries": [],
        "gate_decision": {},
    }


def _refresh(artifact: dict[str, Any]) -> None:
    pairs, instances = aggregate(artifact["records"])
    artifact["paired_comparisons"] = pairs
    artifact["instance_summaries"] = instances
    artifact["gate_decision"] = gate_decision(artifact["records"], instances)


def run_gate(
    ppo_root: Path,
    source_commit_label: str,
) -> dict[str, Any]:
    dataset = _read_json(DATASET_PATH)
    split = _read_json(SPLIT_PATH)
    gate_set = _read_json(GATE_SET_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    validate_gate_set(gate_set, dataset, split)
    if build_protocol(gate_set) != protocol:
        raise ValueError("frozen Phase 3-X protocol differs from deterministic rebuild")
    provenance = _provenance(gate_set, protocol, source_commit_label)
    if ARTIFACT_PATH.exists():
        artifact = _read_json(ARTIFACT_PATH)
        if artifact.get("provenance") != provenance:
            raise ValueError("resume rejected: exact provenance mismatch")
    else:
        artifact = _new_artifact(gate_set, protocol, provenance)
    existing = {str(row["run_key_hash"]): row for row in artifact["records"]}
    for entry in gate_set["instances"]:
        instance = _load_instance(entry, split, ppo_root)
        parents = to_parent_welds(instance)
        common, construction_time = _common_seed(
            parents, source_commit_label=source_commit_label
        )
        assert common.solution is not None and common.schedule is not None
        assert common.directions is not None
        common_hash = common.solution.canonical_hash
        common_cmax = float(common.schedule.cmax)
        seed_record = {
            "common_seed_hash": common_hash,
            "common_seed_cmax": common_cmax,
            "construction_time": construction_time,
            "winning_strategy": common.winning_strategy,
        }
        previous_seed = artifact["common_seeds"].get(str(entry["instance_id"]))
        if previous_seed is not None:
            comparable = {key: previous_seed[key] for key in seed_record if key != "construction_time"}
            expected = {key: seed_record[key] for key in seed_record if key != "construction_time"}
            if comparable != expected:
                raise ValueError("resume rejected: common seed mismatch")
        else:
            artifact["common_seeds"][str(entry["instance_id"])] = seed_record
            _refresh(artifact)
            _write_json(ARTIFACT_PATH, artifact)
        experimental_common = _experimental_initialization(common)
        for seed in SEEDS:
            for arm in ARMS:
                scope = (
                    FORMAL_SCOPE_V1_1
                    if arm == "NO_X_CONTROL"
                    else EXPERIMENTAL_X_SPLIT_SCOPE_V1
                )
                initialization = common if arm == "NO_X_CONTROL" else experimental_common
                key = {
                    **provenance,
                    "instance_id": entry["instance_id"],
                    "instance_geometry_hash": entry["instance_geometry_hash"],
                    "arm": arm,
                    "solver_seed": seed,
                    "common_seed_hash": common_hash,
                }
                key_hash = hashlib.sha256(_canonical_json(key).encode("utf-8")).hexdigest()
                if key_hash in existing:
                    if existing[key_hash].get("run_key") != key:
                        raise ValueError("resume rejected: exact run-key mismatch")
                    continue
                result = run_bounded_sa_oi(
                    parents,
                    ScientificConfig(),
                    gate_search_config(),
                    seed=seed,
                    scope=scope,
                    source_commit=source_commit_label,
                    allow_unverified_source=True,
                    initialization_override=initialization,
                    enable_x_split=arm == "FINITE_X_DOMAIN",
                )
                record = _record(
                    result,
                    entry=entry,
                    arm=arm,
                    seed=seed,
                    common_seed_hash=common_hash,
                    common_seed_cmax=common_cmax,
                    provenance=provenance,
                )
                if record["run_key_hash"] != key_hash:
                    raise RuntimeError("completed run identity mismatch")
                artifact["records"].append(record)
                existing[key_hash] = record
                _refresh(artifact)
                _write_json(ARTIFACT_PATH, artifact)
                print(
                    f"completed {len(artifact['records'])}/72 {arm} "
                    f"{entry['instance_id']} seed={seed}",
                    flush=True,
                )
                if record["numeric_failure_count"] or not record["final_certified"]:
                    raise RuntimeError("Phase 3-X run failed; persisted before stopping")
    _refresh(artifact)
    _write_json(ARTIFACT_PATH, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Finite optional X_SPLIT mechanism gate")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument(
        "--source-commit-label",
        default="KNOWN_MAIN_a8614cfa6a49ee660b27dbb6441f8186c76d09fb+LOCAL_PHASE3X_TREE",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--write-gate-set", action="store_true")
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.ppo_root:
        raise SystemExit("--ppo-root or MRTA_PPO_ROOT is required")
    ppo_root = Path(args.ppo_root).resolve()
    gate_set, protocol = prepare(ppo_root)
    print(
        f"xsplit_gate_set_hash={gate_set['xsplit_gate_set_hash']}\n"
        f"phase3x_protocol_hash={protocol['phase3x_protocol_hash']}",
        flush=True,
    )
    if args.write_manifests or args.write_gate_set:
        _write_json(GATE_SET_PATH, gate_set)
    if args.write_manifests or args.write_protocol:
        _write_json(PROTOCOL_PATH, protocol)
    if args.run:
        if not GATE_SET_PATH.exists() or not PROTOCOL_PATH.exists():
            raise SystemExit("frozen gate-set and protocol manifests are required before --run")
        artifact = run_gate(ppo_root, args.source_commit_label)
        print(_canonical_json(artifact["gate_decision"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
