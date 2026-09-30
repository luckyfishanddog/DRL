from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from mrta_baselines.common import (
    BaselineStatus,
    EvaluatedCandidate,
    canonical_config_hash,
    solution_telemetry,
)
from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns
from mrta_data.phase3_split import ROLE_DEVELOPMENT, assert_solver_access_allowed
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import oriented_endpoints
from mrta_reference.model import ScheduleStatus, ScientificConfig
from mrta_reference.provenance import REPOSITORY_ID, compute_source_tree_hash
from mrta_reference.scheduler import resolve_reference_evaluator
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_reference.solution import block_map, official_metrics
from mrta_search.direction import _fixed_first_dp, _initial_error
from mrta_search.initialization import (
    InitializationAttempt,
    InitializationResult,
    InitializationStatus,
    _construct_rail_serial_bootstrap,
    _patterns,
)
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi


ROOT = Path(__file__).resolve().parents[1]
ALNS_V2_METHOD_ID = "SA_OI_ALNS_INIT_POLICY_V2"
ALNS_V3_NO_TOS_METHOD_ID = "SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_OFF"
ALNS_V3_TOS_METHOD_ID = "SA_OI_ALNS_INIT_POLICY_V3_TWO_OPT_STAR_ON"
HGA_METHOD_ID = "ADAPTED_HGA_V1"
WAG_METHOD_ID = "ADAPTED_WAG_VNS_V1"
SEEDS = (20260928, 20260929, 20260930)
COMMON_SEED = 20260929
CAP_HIT_FIELDS = (
    "vnd_pass_cap_hits",
    "vnd_candidate_cap_hits",
    "initialization_reference_limit_hits",
    "population_survival_events",
    "optional_y_mutations_attempted",
    "optional_y_mutations_accepted",
    "factorial_window_cap_hits",
    "factorial_call_cap_hits",
    "wag_variant_cap_hits",
    "route_combination_cap_hits",
    "move_calls",
    "swap_calls",
    "lns_calls",
    "optional_y_toggle_attempts",
)


def phase3_alns_config(budget: float, *, policy: str = "V2") -> SearchConfig:
    """Phase-3 comparison config: wall clock is primary; iterations are safety only."""
    if policy not in ("V2", "V3_NO_TWO_OPT_STAR", "V3_TWO_OPT_STAR"):
        raise ValueError(f"unsupported ALNS policy: {policy}")
    v3 = policy != "V2"
    return SearchConfig(
        construction_budget=5 if v3 else 4,
        kinit_ref=5 if v3 else 2,
        time_limit=budget,
        max_iterations=100000,
        checkpoints=(5.0, 30.0, 60.0),
        enable_two_opt_star=policy == "V3_TWO_OPT_STAR",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _provenance(
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    source_commit_label: str,
    *,
    diagnostic_set: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "repository_id": REPOSITORY_ID,
        "source_commit": source_commit_label,
        "source_tree_hash": compute_source_tree_hash(ROOT),
        "scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "development_only": True,
        "commit_verified": False,
    }
    if diagnostic_set is not None:
        result["diagnostic_set_hash"] = diagnostic_set["diagnostic_set_hash"]
    return result


def _select_smoke_instances(
    dataset: Mapping[str, Any], split: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    consumed = set(split["development_consumed_workbooks"])
    strata = (
        ("small", 20, 30, 25),
        ("medium", 50, 60, 55),
        ("large", 80, 90, 85),
    )
    selected = []
    used_workbooks = set()
    for tier, lower, upper, target in strata:
        candidates = [
            entry
            for entry in dataset["instances"]
            if entry.get("validation_status") == "VALID"
            and entry.get("duplicate_of") is None
            and entry["relative_path"] in consumed
            and lower <= int(entry["actual_weld_count"]) <= upper
            and entry["relative_path"] not in used_workbooks
        ]
        entry = min(
            candidates,
            key=lambda item: (
                abs(int(item["actual_weld_count"]) - target),
                item["instance_geometry_hash"],
            ),
        )
        chosen = dict(entry)
        chosen["tier"] = tier
        chosen["phase3_role"] = ROLE_DEVELOPMENT
        selected.append(chosen)
        used_workbooks.add(entry["relative_path"])
    assert_solver_access_allowed(
        split,
        tuple(item["relative_path"] for item in selected),
        allowed_roles=(ROLE_DEVELOPMENT,),
    )
    return tuple(selected)


def _entry_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": entry["instance_id"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "relative_path": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "actual_weld_count": entry["actual_weld_count"],
        "phase3_role": entry["phase3_role"],
    }


def _derived_metrics(record: dict[str, Any], common_seed_cmax: float | None) -> None:
    initial = record.get("initial_cmax")
    final = record.get("cmax_at_60")
    record["search_improvement_ratio"] = (
        None
        if initial is None or final is None or initial == 0.0
        else (initial - final) / initial
    )
    record["common_seed_cmax"] = common_seed_cmax
    record["cmax60_over_common_seed"] = (
        None
        if final is None or common_seed_cmax in (None, 0.0)
        else final / common_seed_cmax
    )
    record["absolute_improvement_from_common_seed"] = (
        None if final is None or common_seed_cmax is None else common_seed_cmax - final
    )


def _baseline_record(
    result,
    *,
    entry,
    seed,
    budget,
    provenance,
    comparison_mode,
    experiment_id,
    seed_construction_time=0.0,
    common_seed_cmax=None,
):
    initial = result.initial_candidate
    telemetry = {}
    if initial is not None and initial.directions is not None:
        telemetry = solution_telemetry(
            initial.solution, initial.directions, ScientificConfig()
        )
    record = {
        **provenance,
        **_entry_fields(entry),
        **telemetry,
        "experiment_id": experiment_id,
        "comparison_mode": comparison_mode,
        "method_id": result.method_id,
        "method_config_hash": result.method_config_hash,
        "solver_seed": seed,
        "requested_time_limit_s": budget,
        "termination_reason": result.termination_reason,
        "initialization_status": "SUCCESS" if initial is not None else result.status.value,
        "initial_source": None if initial is None else initial.source,
        "initial_cmax": None if initial is None else initial.metrics.cmax,
        "initial_certified": bool(
            initial is not None
            and initial.certification is not None
            and initial.certification.certified
        ),
        "initial_reference_calls": result.initial_reference_calls,
        "time_to_first_certified": result.time_to_first_certified,
        "best_cmax": None if result.metrics is None else result.metrics.cmax,
        "cmax_at_5": result.checkpoints.get(5.0),
        "cmax_at_30": result.checkpoints.get(30.0),
        "cmax_at_60": result.checkpoints.get(60.0),
        "iterations": result.iterations,
        "candidate_count": result.candidate_count,
        "reference_calls": result.reference_calls,
        "certifier_calls": result.certifier_calls,
        "baseline_DEADLOCK": result.accounting["baseline_deadlock"],
        "recovered": result.accounting["recovered"],
        "remaining_DEADLOCK": result.accounting["remaining_deadlock"],
        "runtime": result.actual_runtime,
        "actual_runtime": result.actual_runtime,
        "overshoot": result.overshoot,
        "initialization_time": result.initialization_time,
        "search_time": result.search_time,
        "search_from_seed_time": result.actual_runtime,
        "seed_construction_time": seed_construction_time,
        "total_with_seed_construction_time": result.actual_runtime + seed_construction_time,
        "scheduler_time": result.accounting["scheduler_time"],
        "local_search_time": result.accounting["local_search_time"],
        "factorial_local_search_time": result.accounting[
            "factorial_local_search_time"
        ],
        "direction_dp_calls": result.accounting["direction_dp_calls"],
        "final_status": result.status.value,
        "final_schedule_status": (
            None if result.schedule is None else result.schedule.status.value
        ),
        "final_certified": result.final_certified,
        "best_source_operator": result.best_source,
        "cap_hits": {key: int(result.accounting[key]) for key in CAP_HIT_FIELDS},
        "diagnostics": list(result.diagnostics),
    }
    _derived_metrics(record, common_seed_cmax)
    return record


def _alns_record(
    result,
    *,
    method_id,
    entry,
    seed,
    budget,
    provenance,
    method_config_hash,
    comparison_mode,
    experiment_id,
    seed_construction_time=0.0,
    common_seed_cmax=None,
):
    references = result.stats.reference_records
    initialization = result.initialization
    telemetry = {}
    if initialization.solution is not None and initialization.directions is not None:
        telemetry = solution_telemetry(
            initialization.solution, initialization.directions, ScientificConfig()
        )
    record = {
        **provenance,
        **_entry_fields(entry),
        **telemetry,
        "experiment_id": experiment_id,
        "comparison_mode": comparison_mode,
        "method_id": method_id,
        "method_config_hash": method_config_hash,
        "solver_seed": seed,
        "requested_time_limit_s": budget,
        "termination_reason": result.termination_reason,
        "initialization_status": initialization.status.value,
        "initial_source": initialization.winning_strategy,
        "initial_cmax": (
            None if initialization.schedule is None else initialization.schedule.cmax
        ),
        "initial_certified": bool(
            initialization.certification and initialization.certification.certified
        ),
        "initial_reference_calls": result.stats.init_reference_calls,
        "time_to_first_certified": (
            result.stats.best_events[0][0] if result.stats.best_events else None
        ),
        "best_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "cmax_at_5": result.anytime.get(5.0, {}).get("cmax"),
        "cmax_at_30": result.anytime.get(30.0, {}).get("cmax"),
        "cmax_at_60": result.anytime.get(60.0, {}).get("cmax"),
        "iterations": result.stats.iterations,
        "candidate_count": result.stats.constructed,
        "reference_calls": len(references),
        "certifier_calls": sum(item["status"] == "FEASIBLE" for item in references) + 1,
        "baseline_DEADLOCK": sum(bool(item["baseline_deadlock"]) for item in references),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in references
        ),
        "remaining_DEADLOCK": sum(
            item["status"] == "DEADLOCK" for item in references
        ),
        "runtime": result.runtime,
        "actual_runtime": result.runtime,
        "overshoot": result.stats.overshoot,
        "initialization_time": result.stats.init_time,
        "search_time": max(0.0, result.runtime - result.stats.init_time),
        "search_from_seed_time": result.runtime,
        "seed_construction_time": seed_construction_time,
        "total_with_seed_construction_time": result.runtime + seed_construction_time,
        "scheduler_time": result.stats.reference_scheduler_time,
        "local_search_time": result.stats.repair_time,
        "factorial_local_search_time": 0.0,
        "direction_dp_calls": result.stats.direction_refinement_calls,
        "final_status": result.status.value,
        "final_schedule_status": (
            None if result.best_schedule is None else result.best_schedule.status.value
        ),
        "final_certified": bool(
            result.final_certification and result.final_certification.certified
        ),
        "best_source_operator": (
            None
            if not result.stats.improvements_by_family
            else max(
                result.stats.improvements_by_family,
                key=result.stats.improvements_by_family.get,
            )
        ),
        "cap_hits": {},
        "move_telemetry": {
            move: {
                "attempted": result.stats.attempted_by_move[move],
                "constructed": result.stats.constructed_by_move[move],
                "cheap_valid": result.stats.cheap_valid_by_move[move],
                "c3": result.stats.c3_by_move[move],
                "c4": result.stats.c4_by_move[move],
                "accepted": result.stats.accepted_by_move[move],
                "best_improvements": result.stats.best_improvement_by_move[move],
            }
            for move in result.stats.attempted_by_move
        },
        "diagnostics": [],
    }
    _derived_metrics(record, common_seed_cmax)
    return record


def _direction_dp_candidates(solution, config):
    """Enumerate bounded DP candidates, then deterministic feasibility fallbacks."""
    blocks = block_map(solution, config)
    routes = tuple(
        tuple(blocks[block_id] for block_id in route.block_ids)
        for route in solution.routes
    )
    active = tuple(robot for robot, route in enumerate(routes) if route)
    cached = {
        (robot, first): _fixed_first_dp(routes[robot], first, config)
        for robot in active
        for first in (0, 1)
    }
    candidates = []
    for bits in itertools.product((0, 1), repeat=len(active)):
        first_by_robot = dict(zip(active, bits))
        points = {
            robot: oriented_endpoints(routes[robot][0], first_by_robot[robot])[0]
            for robot in active
        }
        if _initial_error(points, config) is not None:
            continue
        vectors = []
        total = 0.0
        for robot in range(4):
            if robot not in first_by_robot:
                vectors.append(())
                continue
            cost, vector = cached[(robot, first_by_robot[robot])]
            total += cost
            vectors.append(vector)
        directions = tuple(vectors)
        candidates.append(
            (total, tuple(value for vector in directions for value in vector), directions)
        )
    ordered = [("DIRECTION_DP", item[2]) for item in sorted(candidates)]
    lengths = tuple(len(route) for route in routes)
    fallbacks = (
        ("ALL_ZERO_FEASIBILITY_FALLBACK", tuple(tuple(0 for _ in range(length)) for length in lengths)),
        ("ALL_ONE_FEASIBILITY_FALLBACK", tuple(tuple(1 for _ in range(length)) for length in lengths)),
    )
    seen = {directions for _, directions in ordered}
    for label, directions in fallbacks:
        if directions not in seen:
            ordered.append((label, directions))
            seen.add(directions)
    return tuple(ordered)


def _common_certified_seed(parents, *, entry, budget):
    started = time.perf_counter()
    config = ScientificConfig()
    solution = _construct_rail_serial_bootstrap(parents, _patterns(parents, config), config)
    reference_evaluator = resolve_reference_evaluator(FORMAL_SCOPE_V1_1)
    candidate = None
    reference_calls = 0
    certifier_calls = 0
    statuses = []
    direction_candidates = _direction_dp_candidates(solution, config)
    selected_direction_policy = None
    for direction_policy, directions in direction_candidates:
        schedule = reference_evaluator(
            solution,
            config,
            orientations={robot: directions[robot] for robot in range(4)},
        )
        reference_calls += 1
        statuses.append(schedule.status.value)
        if schedule.status is not ScheduleStatus.FEASIBLE:
            continue
        certification = certify_schedule(
            solution, schedule, config, scope=FORMAL_SCOPE_V1_1
        )
        certifier_calls += 1
        if not certification.certified:
            continue
        candidate = EvaluatedCandidate(
            BaselineStatus.COMPLETED,
            solution,
            directions,
            schedule,
            certification,
            official_metrics(solution, schedule, config),
            "COMMON_CERTIFIED_RAIL_SERIAL_SEED",
            time.perf_counter() - started,
            (),
        )
        selected_direction_policy = direction_policy
        break
    construction_time = time.perf_counter() - started
    if candidate is None:
        raise RuntimeError(
            f"common seed failed certification: {entry['instance_id']}; "
            f"direction-DP statuses={statuses}"
        )
    diagnostic = {
        **_entry_fields(entry),
        **solution_telemetry(solution, candidate.directions, config),
        "seed_source": "RAIL_SERIAL_BOOTSTRAP",
        "common_seed_hash": solution.canonical_hash,
        "common_seed_cmax": candidate.metrics.cmax,
        "seed_certified": True,
        "seed_status": candidate.status.value,
        "seed_construction_time": construction_time,
        "direction_candidate_count": len(direction_candidates),
        "direction_candidates_evaluated": reference_calls,
        "selected_direction_policy": selected_direction_policy,
        "reference_calls": reference_calls,
        "certifier_calls": certifier_calls,
    }
    return candidate, diagnostic


def _alns_seed_override(candidate: EvaluatedCandidate) -> InitializationResult:
    assert candidate.directions is not None
    assert candidate.schedule is not None
    assert candidate.certification is not None
    return InitializationResult(
        InitializationStatus.SUCCESS,
        candidate.solution,
        candidate.directions,
        candidate.schedule,
        candidate.certification,
        (
            InitializationAttempt(
                0,
                candidate.solution,
                None,
                candidate.schedule,
                candidate.certification,
                (),
                False,
                "COMMON_CERTIFIED_RAIL_SERIAL_SEED",
            ),
        ),
        "COMMON_CERTIFIED_RAIL_SERIAL_SEED",
    )


def _run_method(
    method_id,
    parents,
    *,
    entry,
    seed,
    budget,
    provenance,
    comparison_mode,
    experiment_id,
    source_commit_label,
    common_seed=None,
    seed_construction_time=0.0,
):
    common_seed_cmax = None if common_seed is None else common_seed.metrics.cmax
    if method_id in (
        ALNS_V2_METHOD_ID,
        ALNS_V3_NO_TOS_METHOD_ID,
        ALNS_V3_TOS_METHOD_ID,
    ):
        policy = {
            ALNS_V2_METHOD_ID: "V2",
            ALNS_V3_NO_TOS_METHOD_ID: "V3_NO_TWO_OPT_STAR",
            ALNS_V3_TOS_METHOD_ID: "V3_TWO_OPT_STAR",
        }[method_id]
        config = phase3_alns_config(budget, policy=policy)
        result = run_bounded_sa_oi(
            parents,
            ScientificConfig(),
            config,
            seed=seed,
            scope=FORMAL_SCOPE_V1_1,
            source_commit=source_commit_label,
            allow_unverified_source=True,
            formal_result=False,
            initialization_override=(
                None if common_seed is None else _alns_seed_override(common_seed)
            ),
        )
        return _alns_record(
            result,
            method_id=method_id,
            entry=entry,
            seed=seed,
            budget=budget,
            provenance=provenance,
            method_config_hash=canonical_config_hash(config),
            comparison_mode=comparison_mode,
            experiment_id=experiment_id,
            seed_construction_time=seed_construction_time,
            common_seed_cmax=common_seed_cmax,
        )
    if method_id == HGA_METHOD_ID:
        result = run_adapted_hga(
            parents,
            ScientificConfig(),
            AdaptedHGAConfig(),
            seed=seed,
            time_limit=budget,
            scope=FORMAL_SCOPE_V1_1,
            common_seed=common_seed,
        )
    elif method_id == WAG_METHOD_ID:
        result = run_adapted_wag_vns(
            parents,
            ScientificConfig(),
            AdaptedWAGConfig(),
            seed=seed,
            time_limit=budget,
            scope=FORMAL_SCOPE_V1_1,
            common_seed=common_seed,
        )
    else:
        raise ValueError(f"unknown method: {method_id}")
    return _baseline_record(
        result,
        entry=entry,
        seed=seed,
        budget=budget,
        provenance=provenance,
        comparison_mode=comparison_mode,
        experiment_id=experiment_id,
        seed_construction_time=seed_construction_time,
        common_seed_cmax=common_seed_cmax,
    )


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["comparison_mode"],
        record["method_id"],
        record["instance_id"],
        record["solver_seed"],
    )


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for method in sorted({str(record["method_id"]) for record in records}):
        rows = [record for record in records if record["method_id"] == method]
        initial_values = [row["initial_cmax"] for row in rows if row["initial_cmax"] is not None]
        final_values = [row["cmax_at_60"] for row in rows if row["cmax_at_60"] is not None]
        first_values = [
            row["time_to_first_certified"]
            for row in rows
            if row["time_to_first_certified"] is not None
        ]
        result[method] = {
            "run_count": len(rows),
            "certified_count": sum(bool(row["final_certified"]) for row in rows),
            "mean_initial_cmax": None if not initial_values else statistics.fmean(initial_values),
            "median_initial_cmax": None if not initial_values else statistics.median(initial_values),
            "mean_cmax_at_60": None if not final_values else statistics.fmean(final_values),
            "median_cmax_at_60": None if not final_values else statistics.median(final_values),
            "median_time_to_first_certified": (
                None if not first_values else statistics.median(first_values)
            ),
            "total_reference_calls": sum(row["reference_calls"] for row in rows),
            "mean_improvement_per_reference_call": statistics.fmean(
                (row["initial_cmax"] - row["cmax_at_60"])
                / max(1, row["reference_calls"])
                for row in rows
                if row["initial_cmax"] is not None and row["cmax_at_60"] is not None
            ) if final_values else None,
            "termination_reasons": {
                reason: sum(row["termination_reason"] == reason for row in rows)
                for reason in sorted({row["termination_reason"] for row in rows})
            },
        }
    return result


def _initial_hash_analysis(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for instance_id in sorted({row["instance_id"] for row in records}):
        for seed in SEEDS:
            hga = next(
                row
                for row in records
                if row["instance_id"] == instance_id
                and row["solver_seed"] == seed
                and row["method_id"] == HGA_METHOD_ID
            )
            wag = next(
                row
                for row in records
                if row["instance_id"] == instance_id
                and row["solver_seed"] == seed
                and row["method_id"] == WAG_METHOD_ID
            )
            if hga["initial_solution_hash"] == wag["initial_solution_hash"]:
                finding = "A_CANONICAL_SOLUTION_HASH_IDENTICAL"
            elif (
                hga["initial_patterns"] == wag["initial_patterns"]
                and [set(route) for route in hga["initial_routes"]]
                == [set(route) for route in wag["initial_routes"]]
            ):
                finding = "B_ASSIGNMENT_IDENTICAL_ROUTES_DIFFER"
            elif abs(hga["initial_cmax"] - wag["initial_cmax"]) <= 1.0e-9:
                finding = "C_DIFFERENT_SOLUTION_EQUAL_CMAX"
            else:
                finding = "DIFFERENT_SOLUTION_AND_CMAX"
            result.append(
                {
                    "instance_id": instance_id,
                    "solver_seed": seed,
                    "finding": finding,
                    "shared_initializer_detected": False,
                    "hga": {key: hga[key] for key in (
                        "initial_solution_hash", "initial_source", "initial_patterns",
                        "initial_robot_block_counts", "initial_robot_process_loads",
                        "initial_route_hashes", "initial_directions", "initial_cmax",
                    )},
                    "wag": {key: wag[key] for key in (
                        "initial_solution_hash", "initial_source", "initial_patterns",
                        "initial_robot_block_counts", "initial_robot_process_loads",
                        "initial_route_hashes", "initial_directions", "initial_cmax",
                    )},
                }
            )
    return result


def _apply_initial_gap(records: list[dict[str, Any]]) -> None:
    for row in records:
        paired = [
            item for item in records
            if item["instance_id"] == row["instance_id"]
            and item["solver_seed"] == row["solver_seed"]
        ]
        available = [item["initial_cmax"] for item in paired if item["initial_cmax"] is not None]
        if row["initial_cmax"] is None or not available:
            row["paired_best_native_initial_cmax"] = min(available) if available else None
            row["initial_gap_to_common_best"] = None
            row["initial_gap_ratio_to_common_best"] = None
            continue
        common_best = min(available)
        row["paired_best_native_initial_cmax"] = common_best
        row["initial_gap_to_common_best"] = row["initial_cmax"] - common_best
        row["initial_gap_ratio_to_common_best"] = row["initial_cmax"] / common_best - 1.0


def _decision(common_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method = {
        method: [row for row in common_records if row["method_id"] == method]
        for method in (ALNS_V2_METHOD_ID, HGA_METHOD_ID, WAG_METHOD_ID)
    }
    medians = {
        method: statistics.median(row["cmax_at_60"] for row in rows)
        for method, rows in by_method.items()
    }
    median_ratio = medians[ALNS_V2_METHOD_ID] / min(
        medians[HGA_METHOD_ID], medians[WAG_METHOD_ID]
    )
    instance_rows = []
    within_count = 0
    for instance_id in sorted({row["instance_id"] for row in common_records}):
        values = {
            row["method_id"]: row["cmax_at_60"]
            for row in common_records if row["instance_id"] == instance_id
        }
        ratio = values[ALNS_V2_METHOD_ID] / min(values[HGA_METHOD_ID], values[WAG_METHOD_ID])
        within = ratio <= 1.10 + 1.0e-12
        within_count += int(within)
        instance_rows.append({
            "instance_id": instance_id,
            "cmax_at_60": values,
            "alns_to_best_baseline_ratio": ratio,
            "within_10_percent": within,
        })
    case_a = median_ratio <= 1.10 + 1.0e-12 or within_count >= 4
    return {
        "rule_id": "PHASE3_2A_PREDECLARED_10_PERCENT_RULE",
        "method_median_cmax_at_60": medians,
        "alns_median_to_best_baseline_median_ratio": median_ratio,
        "instances_within_10_percent": within_count,
        "instance_count": 6,
        "paired_instance_results": instance_rows,
        "case": "CASE_A_INITIALIZATION_DOMINATED" if case_a else "CASE_B_SEARCH_MECHANISM_GAP",
        "production_correction": (
            "INIT_POLICY_V3_ONLY" if case_a
            else "INIT_POLICY_V3_PLUS_ONE_TWO_OPT_STAR_OPERATOR"
        ),
    }


def _run_fairness_pre(args, dataset, split, ppo_root) -> int:
    diagnostic = _load_json(ROOT / args.diagnostic_set)
    selected = tuple(dict(item) for item in diagnostic["instances"])
    assert_solver_access_allowed(
        split, tuple(item["relative_path"] for item in selected),
        allowed_roles=(ROLE_DEVELOPMENT,),
    )
    provenance = _provenance(
        dataset, split, args.source_commit_label, diagnostic_set=diagnostic
    )
    output = ROOT / args.output
    if args.resume and output.exists():
        payload = _load_json(output)
        if payload.get("pre_fix_provenance") != provenance:
            raise ValueError("resume provenance does not match current pre-fix source")
    else:
        payload = {
            "artifact_id": "PHASE3_FAIRNESS_DIAGNOSTIC_V1",
            "experiment_id": "PHASE3_2A_FAIRNESS_PRE_FIX_V1",
            "diagnostic_set": diagnostic,
            "pre_fix_provenance": provenance,
            "native_records": [],
            "common_seed_construction": [],
            "common_seed_records": [],
            "decision_rule": None,
            "pre_fix_summary": None,
            "initial_hash_analysis": [],
            "post_fix_confirmation": [],
        }
        _write_json(output, payload)

    native_keys = {_record_key(row) for row in payload["native_records"]}
    common_keys = {_record_key(row) for row in payload["common_seed_records"]}
    seed_by_instance = {
        row["instance_id"]: row for row in payload["common_seed_construction"]
    }
    methods = (ALNS_V2_METHOD_ID, HGA_METHOD_ID, WAG_METHOD_ID)
    budget = 60.0
    for entry in selected:
        instance = load_ppo_platform_instance(
            ppo_root / entry["relative_path"], entry["sheet_name"],
            ppo_root=ppo_root, instance_id=entry["instance_id"],
        )
        parents = to_parent_welds(instance)
        for seed in SEEDS:
            for method in methods:
                key = ("NATIVE", method, entry["instance_id"], seed)
                if key in native_keys:
                    continue
                record = _run_method(
                    method, parents, entry=entry, seed=seed, budget=budget,
                    provenance=provenance, comparison_mode="NATIVE",
                    experiment_id=payload["experiment_id"],
                    source_commit_label=args.source_commit_label,
                )
                payload["native_records"].append(record)
                native_keys.add(key)
                _write_json(output, payload)
                print(json.dumps({
                    "completed": len(payload["native_records"]), "expected": 54,
                    "mode": "NATIVE", "method": method,
                    "instance": entry["instance_id"], "seed": seed,
                    "termination": record["termination_reason"],
                }, ensure_ascii=False), flush=True)

        common_seed, generated_seed_record = _common_certified_seed(
            parents, entry=entry, budget=budget
        )
        if entry["instance_id"] not in seed_by_instance:
            generated_seed_record.update(provenance)
            payload["common_seed_construction"].append(generated_seed_record)
            seed_by_instance[entry["instance_id"]] = generated_seed_record
            _write_json(output, payload)
        seed_record = seed_by_instance[entry["instance_id"]]
        for method in methods:
            key = ("COMMON_SEED_DIAGNOSTIC", method, entry["instance_id"], COMMON_SEED)
            if key in common_keys:
                continue
            record = _run_method(
                method, parents, entry=entry, seed=COMMON_SEED, budget=budget,
                provenance=provenance, comparison_mode="COMMON_SEED_DIAGNOSTIC",
                experiment_id=payload["experiment_id"],
                source_commit_label=args.source_commit_label,
                common_seed=common_seed,
                seed_construction_time=seed_record["seed_construction_time"],
            )
            payload["common_seed_records"].append(record)
            common_keys.add(key)
            _write_json(output, payload)
            print(json.dumps({
                "completed": len(payload["common_seed_records"]), "expected": 18,
                "mode": "COMMON_SEED_DIAGNOSTIC", "method": method,
                "instance": entry["instance_id"], "seed": COMMON_SEED,
                "termination": record["termination_reason"],
            }, ensure_ascii=False), flush=True)

    if len(payload["native_records"]) != 54 or len(payload["common_seed_records"]) != 18:
        raise RuntimeError("fairness pre-diagnostic is incomplete")
    _apply_initial_gap(payload["native_records"])
    payload["pre_fix_summary"] = {
        "native": _summary(payload["native_records"]),
        "common_seed": _summary(payload["common_seed_records"]),
    }
    payload["initial_hash_analysis"] = _initial_hash_analysis(payload["native_records"])
    payload["decision_rule"] = _decision(payload["common_seed_records"])
    payload["pre_fix_complete"] = True
    _write_json(output, payload)
    print(json.dumps({
        "output": str(output), "native_records": 54,
        "common_seed_records": 18,
        "decision": payload["decision_rule"]["case"],
    }, ensure_ascii=False, indent=2), flush=True)
    all_records = payload["native_records"] + payload["common_seed_records"]
    return 0 if all(row["final_certified"] for row in all_records) else 2


def _run_baseline_smoke(args, dataset, split, ppo_root) -> int:
    selected = _select_smoke_instances(dataset, split)
    provenance = _provenance(dataset, split, args.source_commit_label)
    records = []
    common_seed_diagnostics = []
    methods = (ALNS_V2_METHOD_ID, HGA_METHOD_ID, WAG_METHOD_ID)
    for entry in selected:
        instance = load_ppo_platform_instance(
            ppo_root / entry["relative_path"], entry["sheet_name"],
            ppo_root=ppo_root, instance_id=entry["instance_id"],
        )
        parents = to_parent_welds(instance)
        _, seed_record = _common_certified_seed(parents, entry=entry, budget=args.budget)
        common_seed_diagnostics.append(seed_record)
        for seed in SEEDS:
            for method in methods:
                records.append(_run_method(
                    method, parents, entry=entry, seed=seed, budget=args.budget,
                    provenance=provenance, comparison_mode="NATIVE",
                    experiment_id="PHASE3_BASELINE_SMOKE_V1",
                    source_commit_label=args.source_commit_label,
                ))
    payload = {
        "smoke_id": "PHASE3_BASELINE_SMOKE_V1",
        "solver_seeds": list(SEEDS), "selected_instances": list(selected),
        "provenance": provenance,
        "common_certified_seed_diagnostics": common_seed_diagnostics,
        "records": records,
    }
    _write_json(ROOT / args.output, payload)
    return 0 if all(record["final_certified"] for record in records) else 2


def _run_fairness_post(args, dataset, split, ppo_root) -> int:
    output = ROOT / args.output
    payload = _load_json(output)
    if not payload.get("pre_fix_complete"):
        raise ValueError("post-fix confirmation requires a complete pre-fix diagnostic")
    decision = payload["decision_rule"]["case"]
    if decision != "CASE_B_SEARCH_MECHANISM_GAP":
        raise ValueError(f"this implementation expects frozen CASE B, got {decision}")
    diagnostic = payload["diagnostic_set"]
    selected = tuple(dict(item) for item in diagnostic["instances"])
    assert_solver_access_allowed(
        split,
        tuple(item["relative_path"] for item in selected),
        allowed_roles=(ROLE_DEVELOPMENT,),
    )
    provenance = _provenance(
        dataset, split, args.source_commit_label, diagnostic_set=diagnostic
    )
    if args.resume and payload.get("post_fix_provenance") not in (None, provenance):
        raise ValueError("resume provenance does not match current post-fix source")
    payload["post_fix_provenance"] = provenance
    payload["post_fix_experiment_id"] = "PHASE3_2A_POST_FIX_CONFIRMATION_V1"
    methods = (
        ALNS_V2_METHOD_ID,
        ALNS_V3_NO_TOS_METHOD_ID,
        ALNS_V3_TOS_METHOD_ID,
        HGA_METHOD_ID,
        WAG_METHOD_ID,
    )
    existing = {_record_key(row) for row in payload["post_fix_confirmation"]}
    budget = 60.0
    for entry in selected:
        instance = load_ppo_platform_instance(
            ppo_root / entry["relative_path"],
            entry["sheet_name"],
            ppo_root=ppo_root,
            instance_id=entry["instance_id"],
        )
        parents = to_parent_welds(instance)
        for method in methods:
            key = ("NATIVE", method, entry["instance_id"], COMMON_SEED)
            if key in existing:
                continue
            record = _run_method(
                method,
                parents,
                entry=entry,
                seed=COMMON_SEED,
                budget=budget,
                provenance=provenance,
                comparison_mode="NATIVE",
                experiment_id=payload["post_fix_experiment_id"],
                source_commit_label=args.source_commit_label,
            )
            payload["post_fix_confirmation"].append(record)
            existing.add(key)
            _write_json(output, payload)
            print(
                json.dumps(
                    {
                        "completed": len(payload["post_fix_confirmation"]),
                        "expected": 30,
                        "mode": "POST_FIX_CONFIRMATION",
                        "method": method,
                        "instance": entry["instance_id"],
                        "termination": record["termination_reason"],
                        "certified": record["final_certified"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if len(payload["post_fix_confirmation"]) != 30:
        raise RuntimeError("post-fix confirmation is incomplete")
    _apply_initial_gap(payload["post_fix_confirmation"])
    payload["post_fix_summary"] = _summary(payload["post_fix_confirmation"])
    payload["production_definition"] = {
        "method_id": ALNS_V3_TOS_METHOD_ID,
        "initialization_policy": "SA_OI_ALNS_INIT_POLICY_V3",
        "initialization_strategy_added": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
        "max_initialization_reference_calls": 10,
        "enable_two_opt_star": True,
        "additional_search_operators": ["TWO_OPT_STAR"],
    }
    payload["post_fix_complete"] = True
    _write_json(output, payload)
    required = [
        row
        for row in payload["post_fix_confirmation"]
        if row["method_id"] != ALNS_V2_METHOD_ID
    ]
    return 0 if all(row["final_certified"] for row in required) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 comparison diagnostics")
    parser.add_argument(
        "--mode", choices=("baseline-smoke", "fairness-pre", "fairness-post"),
        default="baseline-smoke",
    )
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--source-commit-label", required=True)
    parser.add_argument(
        "--diagnostic-set",
        default="data/manifests/PPO_PHASE3_DIAGNOSTIC_SET_V1.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    if args.output is None:
        args.output = (
            "data/development/phase3_fairness_diagnostic_v1.json"
            if args.mode in ("fairness-pre", "fairness-post")
            else "data/development/phase3_baseline_smoke_v1.json"
        )
    ppo_root = Path(args.ppo_root).resolve()
    dataset = _load_json(ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json")
    split = _load_json(ROOT / "data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json")
    if args.mode == "fairness-pre":
        return _run_fairness_pre(args, dataset, split, ppo_root)
    if args.mode == "fairness-post":
        return _run_fairness_post(args, dataset, split, ppo_root)
    return _run_baseline_smoke(args, dataset, split, ppo_root)


if __name__ == "__main__":
    raise SystemExit(main())
