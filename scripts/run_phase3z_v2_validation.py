"""Frozen V2 validation orchestration; never changes a solver or opens other roles.

Run with the project root on PYTHONPATH: python -B -m scripts.run_phase3z_v2_validation.
--prepare writes metadata only. Formal execution requires a clean, standalone DRL
checkout. Generated protocol/results are not scientific source; keep their exact
paths outside Git tracking (local .git/info/exclude), not source directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from functools import wraps

from mrta_baselines.common import canonical_config_hash, solution_telemetry
from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns
from mrta_data.phase3_split import ROLE_V2_VALIDATION, assert_v2_solver_access_allowed
from mrta_data.ppo_instances import (
    instance_geometry_hash, load_ppo_platform_instance, to_parent_welds,
)
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import build_legal_pattern_catalog, pattern_catalog_hash
from mrta_reference.model import OperationKind, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.provenance import (
    REPOSITORY_ID, SourceProvenanceError, compute_source_tree_hash, resolve_source_provenance,
)
from mrta_reference.scope import (
    ACTIVE_FORMAL_SCOPE, FORMAL_SCOPE_V1, FORMAL_SCOPE_V1_1, FORMAL_SCOPE_V2,
)
from mrta_search.pipeline import SearchConfig, run_sa_oi_alns_v2

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/manifests/PHASE3Z_V2_VALIDATION_PROTOCOL_V1.json"
ARTIFACT_PATH = ROOT / "data/validation/phase3z_v2_common_model_validation_v1.json"
HANDOFF_PATH = ROOT / "docs/PHASE3Z_V2_COMMON_MODEL_VALIDATION_20261006.md"
ROLES_PATH = ROOT / "data/manifests/PPO_V2_DATA_ROLES_V1.json"
VALIDATION_PATH = ROOT / "data/manifests/PPO_V2_VALIDATION_SET_V1.json"
YR_PATH = ROOT / "data/manifests/PHASE3YR_V2_ACCESS_PROTOCOL_V1.json"
SEED_SET_ID = "PHASE3Z_VALIDATION_SEEDS_V1"
SEEDS = (20261011, 20261012, 20261013, 20261014, 20261015)
METHODS = ("SA_OI_ALNS_V2", "ADAPTED_HGA_V2", "ADAPTED_WAG_VNS_V2")
TIME_LIMIT = 60.0
CHECKPOINTS = (5.0, 30.0, 60.0)
FROZEN = {
    "FORMAL_SCOPE_V1": "8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9",
    "FORMAL_SCOPE_V1_1": "5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc",
    "FORMAL_SCOPE_V2": "16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599",
    "v2_data_roles_hash": "7a80372eb072b10da3d8044eb8bb330a29ede93fe16c806b1aa0d58bee968a5a",
    "v2_validation_set_hash": "2989d8fe15330883a617504d4cc51a6a6cfc547aa9c9813853b6df8fcb2245fe",
    "phase3yr_protocol_hash": "e2206a3d20a2dddf53912920abbf4ca62844a669db322a7cce51ce3655b99876",
}
COMPETITION_GATE = {"median_R_max": 1.10, "R_threshold": 1.10,
                    "minimum_count": 8, "tier_median_R_max": 1.15}
CAP_FIELDS = ("vnd_candidate_cap_hits", "vnd_pass_cap_hits",
              "initialization_reference_limit_hits", "population_survival_events",
              "factorial_window_cap_hits", "factorial_call_cap_hits",
              "wag_variant_cap_hits", "route_combination_cap_hits")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def protocol_hash(payload):
    return digest({k: v for k, v in payload.items() if k != "phase3z_protocol_hash"})


def json_ready(value):
    # dataclasses.asdict reconstructs Counter from pairs incorrectly. Preserve
    # its original key/value mapping rather than silently corrupting telemetry.
    if is_dataclass(value):
        return {field.name: json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def atomic_json(path, payload):
    """Same-directory fsync + replace; interruptions never leave a partial result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def method_configs():
    return {METHODS[0]: SearchConfig(construction_budget=5, kinit_ref=5,
                                    max_iterations=100000, time_limit=TIME_LIMIT,
                                    checkpoints=CHECKPOINTS, enable_two_opt_star=False),
            METHODS[1]: AdaptedHGAConfig(), METHODS[2]: AdaptedWAGConfig()}


def frozen_metadata():
    """Manifest metadata only; no workbook parsing, including validation workbooks."""
    roles, validation, yr = map(read_json, (ROLES_PATH, VALIDATION_PATH, YR_PATH))
    for payload, field in ((roles, "v2_data_roles_hash"),
                           (validation, "v2_validation_set_hash"),
                           (yr, "phase3yr_protocol_hash")):
        if payload[field] != FROZEN[field] or digest({k: v for k, v in payload.items()
                                                     if k != field}) != FROZEN[field]:
            raise ValueError(f"frozen manifest changed: {field}")
    for scope in (FORMAL_SCOPE_V1, FORMAL_SCOPE_V1_1, FORMAL_SCOPE_V2):
        if scope.scope_hash != FROZEN[scope.scope_id]:
            raise ValueError(f"frozen scope changed: {scope.scope_id}")
    if ACTIVE_FORMAL_SCOPE != FORMAL_SCOPE_V2:
        raise ValueError("ACTIVE scope must already be V2")
    for relative, expected in yr["protected_file_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"YR protected source/history changed: {relative}")
    instances = validation["instances"]
    if len(instances) != 12 or Counter(e["tier"] for e in instances) != {
            "SMALL": 4, "MEDIUM": 4, "LARGE": 4}:
        raise ValueError("validation population changed")
    assert_v2_solver_access_allowed(roles, [e["relative_path"] for e in instances],
                                    allowed_roles=(ROLE_V2_VALIDATION,))
    yr_configs = {m["method_id"]: m["method_config_hash"] for m in yr["methods"]}
    for method, config in method_configs().items():
        comparable = config
        if method == METHODS[0]:
            from dataclasses import replace
            comparable = replace(config, time_limit=30.0, checkpoints=(5.0, 15.0, 30.0))
        if canonical_config_hash(comparable) != yr_configs[method]:
            raise ValueError(f"frozen YR parameters changed: {method}")
    return roles, validation, yr


def run_order(instances):
    result = []
    for index, entry in enumerate(instances):
        for seed_index, seed in enumerate(SEEDS):
            offset = (index + seed_index) % 3
            for method in METHODS[offset:] + METHODS[:offset]:
                result.append({"instance_id": entry["instance_id"],
                               "method_id": method, "solver_seed": seed})
    return result


def build_protocol(*, require_verified=True):
    _, validation, _ = frozen_metadata()
    try:
        provenance = resolve_source_provenance(ROOT)
        provenance.require_formal_result()
        verify_imported_source()
        source = asdict(provenance)
        source["status"] = "VERIFIED_CLEAN"
    except SourceProvenanceError as error:
        if require_verified:
            raise
        source = {"repository_id": REPOSITORY_ID, "source_commit": None,
                  "source_tree_hash": compute_source_tree_hash(ROOT),
                  "commit_verified": False, "worktree_dirty": True,
                  "status": "BLOCKED_BY_PROVENANCE", "diagnostic": str(error)}
    configs = method_configs()
    payload = {
        "protocol_id": "PHASE3Z_V2_VALIDATION_PROTOCOL_V1",
        "protocol_status": "FROZEN" if source["commit_verified"] else "BLOCKED_BY_PROVENANCE",
        "provenance": source, "frozen_identities": FROZEN,
        "scientific_config": asdict(ScientificConfig()),
        "scientific_config_hash": ScientificConfig().scientific_hash,
        "methods": {m: {"config": asdict(c), "method_config_hash": canonical_config_hash(c)}
                    for m, c in configs.items()},
        "access_policies": {"ALNS_C2": "V2_C2_PATTERN_FAMILY_STRATIFIED_V1",
                            "ALNS_C4": "V2_C4_FAMILY_EXPLORE_EXPLOIT_V1",
                            "WAG": "V2_PER_ITERATION_PATTERN_FIRST_SEEDED_FAMILY_V1"},
        "instances": validation["instances"], "seed_set_id": SEED_SET_ID,
        "seeds": list(SEEDS), "time_limit": TIME_LIMIT, "checkpoints": list(CHECKPOINTS),
        "run_count": 180, "execution_mode": "SEQUENTIAL_SINGLE_TRAJECTORY",
        "run_order_policy": "instance-major, seed-major; methods rotated (instance_index+seed_index)%3",
        "run_order": run_order(validation["instances"]),
        "checkpoint_semantics": "minimum certified incumbent completed at or before deadline; null if absent; no backfill",
        "overshoot_policy": "nonpreemptive completion allowed; final_cmax diagnostic only; main metric Cmax@60",
        "initialization_policy": "native initialization included in each 60s solver clock; no injected seed",
        "resume_policy": "exact run_key including protocol/scope/validation/geometry/method/config/seed/source/HEAD; atomic persist each run",
        "aggregation_policy": "five distinct frozen seeds all certified Cmax@60 or INCOMPLETE_CERTIFICATION; early medians require five values too",
        "competition_gate": COMPETITION_GATE,
        "ratio_formula": "ALNS five-seed median Cmax@60 / min(HGA median Cmax@60, WAG median Cmax@60)",
        "allowed_role": ROLE_V2_VALIDATION,
        "prohibited_roles": ["V2_TRAIN_POOL", "V2_MODEL_DEVELOPMENT_CONSUMED", "ID_TEST_SEALED"],
        "role_authority": "PPO_V2_DATA_ROLES_V1, never filesystem folder names",
        "algorithm_source_policy": "YR scientific source and parameters immutable; no validation-driven tuning",
    }
    payload["phase3z_protocol_hash"] = protocol_hash(payload)
    return json_ready(payload)


def verify_imported_source():
    """A clean checkout is insufficient if an editable install imports another tree."""
    import mrta_search.pipeline as pipeline
    import mrta_baselines.hga as hga
    import mrta_baselines.wag_vns as wag
    import mrta_reference.certifier as certifier
    import mrta_data.ppo_instances as data
    source_root = (ROOT / "src").resolve()
    for module in (pipeline, hga, wag, certifier, data):
        if not Path(module.__file__).resolve().is_relative_to(source_root):
            raise SourceProvenanceError(f"imported scientific module is outside executing checkout: {module.__name__}")


@contextmanager
def direction_call_observer():
    """Count existing per-route DP invocations without changing results or order.

    The observer executes inside the native solver clock; its small bookkeeping
    cost is not free and applies identically to all three frozen methods.
    """
    import mrta_search.direction as direction
    original = direction._fixed_first_dp
    counter = {"route_dp_calls": 0}
    @wraps(original)
    def observed(*args, **kwargs):
        counter["route_dp_calls"] += 1
        return original(*args, **kwargs)
    direction._fixed_first_dp = observed
    try:
        yield counter
    finally:
        direction._fixed_first_dp = original


def verify_protocol(protocol):
    if protocol.get("protocol_status") != "FROZEN":
        raise SourceProvenanceError("BLOCKED_BY_PROVENANCE: protocol is not formally frozen")
    if protocol_hash(protocol) != protocol["phase3z_protocol_hash"]:
        raise ValueError("protocol integrity mismatch")
    if protocol != build_protocol():
        raise ValueError("protocol/source/config/manifest identity differs from frozen execution")


def run_key(protocol, entry, method, seed):
    return {"phase3z_protocol_hash": protocol["phase3z_protocol_hash"],
            "scope_hash": FROZEN["FORMAL_SCOPE_V2"],
            "v2_validation_set_hash": FROZEN["v2_validation_set_hash"],
            "instance_id": entry["instance_id"],
            "instance_geometry_hash": entry["instance_geometry_hash"],
            "method_id": method, "method_config_hash": protocol["methods"][method]["method_config_hash"],
            "solver_seed": seed, "source_tree_hash": protocol["provenance"]["source_tree_hash"],
            "source_commit": protocol["provenance"]["source_commit"]}


def validate_resume(artifact, protocol):
    if artifact.get("protocol") != protocol:
        raise ValueError("resume protocol differs")
    entries = {e["instance_id"]: e for e in protocol["instances"]}
    expected = {digest(run_key(protocol, entries[o["instance_id"]], o["method_id"], o["solver_seed"]))
                for o in protocol["run_order"]}
    completed = {}
    for row in artifact.get("records", []):
        key_hash = digest(row["run_key"])
        if key_hash not in expected or row.get("run_key_hash") != key_hash or key_hash in completed:
            raise ValueError("invalid, duplicate, or mismatched resume run key")
        if row.get("execution_status") != "COMPLETED":
            raise ValueError("completed records cannot contain a partial run")
        # Never trust only the supplied key while accepting conflicting record fields.
        for field, value in row["run_key"].items():
            if field in row and row[field] != value:
                raise ValueError(f"resume record identity differs: {field}")
        completed[key_hash] = row
    return completed


def load_validation_parents(roles, entry, ppo_root, *, loader=None):
    assert_v2_solver_access_allowed(roles, [entry["relative_path"]],
                                    allowed_roles=(ROLE_V2_VALIDATION,))
    loader = loader or load_ppo_platform_instance
    instance = loader(Path(ppo_root) / entry["relative_path"], entry["sheet_name"],
                      ppo_root=Path(ppo_root), instance_id=entry["instance_id"])
    if instance.raw_file_sha256 != entry["workbook_sha256"]:
        raise ValueError("frozen workbook bytes differ")
    if instance_geometry_hash(instance) != entry["instance_geometry_hash"]:
        raise ValueError("frozen geometry differs")
    parents = to_parent_welds(instance)
    if len(parents) != entry["N"]:
        raise ValueError("frozen weld count differs")
    return parents


def checkpoint_values(events):
    return {f"cmax_at_{int(deadline)}": min((c for t, c in events if t <= deadline), default=None)
            for deadline in CHECKPOINTS}


def final_structure(solution, directions, schedule):
    if solution is None or directions is None or schedule is None:
        return {"final_structure": None}
    telemetry = solution_telemetry(solution, directions, ScientificConfig())
    patterns = solution.patterns
    waits = [sum(o.duration for o in schedule.operations
                 if o.robot_id == robot and o.kind is OperationKind.WAIT) for robot in range(4)]
    return {"final_structure": {
        "whole_parent_count": sum(p.kind is SplitKind.WHOLE for p in patterns),
        "mandatory_y_count": sum(p.kind is SplitKind.Y_SPLIT and p.mandatory for p in patterns),
        "optional_y_count": sum(p.kind is SplitKind.Y_SPLIT and not p.mandatory for p in patterns),
        "x_count": sum(p.kind is SplitKind.X_SPLIT for p in patterns),
        "pattern_ids": [p.pattern_id for p in patterns],
        "patterns": telemetry["initial_patterns"],
        "robot_block_counts": telemetry["initial_robot_block_counts"],
        "robot_process_loads": telemetry["initial_robot_process_loads"],
        "robot_wait": waits, "robot_completion": list(schedule.robot_completion),
        "empty_travel": telemetry["initial_total_empty_travel_proxy"],
        "makespan_robot": max(range(4), key=lambda r: schedule.robot_completion[r]),
        "routes": telemetry["initial_routes"], "directions": telemetry["initial_directions"],
    }, "final_x_parent_count": sum(p.kind is SplitKind.X_SPLIT for p in patterns),
        "final_x_pattern_ids": [p.pattern_id for p in patterns if p.kind is SplitKind.X_SPLIT],
        "final_y_parent_count": sum(p.kind is SplitKind.Y_SPLIT for p in patterns),
        "final_y_pattern_ids": [p.pattern_id for p in patterns if p.kind is SplitKind.Y_SPLIT]}


def run_method(protocol, entry, method, seed, parents):
    configs = method_configs()
    provenance = resolve_source_provenance(ROOT)
    provenance.require_formal_result()
    if asdict(provenance) != {k: v for k, v in protocol["provenance"].items() if k != "status"}:
        raise ValueError("source changed immediately before solver execution")
    if method == METHODS[0]:
        with direction_call_observer() as direction_count:
            result = run_sa_oi_alns_v2(parents, ScientificConfig(), configs[method], seed=seed,
                                      source_provenance=provenance, formal_result=True)
        stats = result.stats
        solution, directions, schedule = result.best_solution, result.best_directions, result.best_schedule
        events = list(stats.best_events)
        references = stats.reference_records
        numeric = sum(r["status"] == "NUMERIC_FAILURE" for r in references)
        numeric += int(result.status.value == "NUMERIC_FAILURE")
        metrics = {
            "initialization_time": stats.init_time, "iterations": stats.iterations,
            "candidate_count": stats.constructed, "reference_calls": len(references),
            "certifier_calls": sum(r["status"] == "FEASIBLE" for r in references) + int(schedule is not None),
            "certifier_calls_semantics": "feasible reference count plus final certification; see raw stats for refinements",
            "direction_dp_calls": direction_count["route_dp_calls"],
            "direction_dp_calls_semantics": "observed native per-route fixed-first DP invocations, including initialization; bookkeeping included in native clock",
            "direction_c3_candidates": stats.kdp_count,
            "scheduler_time": stats.reference_scheduler_time, "direction_time": stats.direction_dp_time,
            "construction_time": stats.candidate_generation_time, "repair_time": stats.repair_time,
            "local_search_time": stats.repair_time, "actual_runtime": result.runtime,
            "overshoot": stats.overshoot,
            "DEADLOCK": sum(bool(r["baseline_deadlock"]) for r in references),
            "recovered": sum(bool(r["baseline_deadlock"]) and r["status"] == "FEASIBLE" for r in references),
            "remaining_DEADLOCK": sum(r["status"] == "DEADLOCK" for r in references),
            "numeric_failure_count": numeric,
            "intermediate_certifier_mismatch_count": sum(any("failed certification" in str(d) for d in r["diagnostics"]) for r in references),
            "cap_hit_telemetry": {field: None for field in CAP_FIELDS},
            "decision_family_funnel": {family: dict(c) for family, c in stats.decision_family_funnel.items()},
            "raw_accounting": json_ready(stats),
        }
        for prefix in ("x", "y"):
            for output, field in (("proposals", "generated"), ("constructed", "cheap_feasible"),
                                  ("direction_evaluated", "c3"), ("reference_evaluated", "reference_evaluated"),
                                  ("certified", "certified"), ("accepted", "accepted")):
                metrics[f"{prefix}_pattern_{output}"] = getattr(stats, f"{prefix}_pattern_candidates_{field}")
            metrics[f"{prefix}_pattern_global_best_updates"] = getattr(stats, f"{prefix}_pattern_global_best_updates")
    else:
        runner = run_adapted_hga if method == METHODS[1] else run_adapted_wag_vns
        kwargs = {} if method == METHODS[1] else {"pattern_access_policy": True}
        with direction_call_observer() as direction_count:
            result = runner(parents, ScientificConfig(), configs[method], seed=seed,
                            time_limit=TIME_LIMIT, checkpoints=CHECKPOINTS,
                            scope=FORMAL_SCOPE_V2, method_id=method, **kwargs)
        if result.method_config_hash != protocol["methods"][method]["method_config_hash"]:
            raise ValueError("executed method config hash differs")
        solution, directions, schedule = result.solution, result.directions, result.schedule
        events = [(e.elapsed, e.cmax) for e in result.best_events]
        account = dict(result.accounting)
        metrics = {**{f: account[f] for f in ("direction_dp_calls", "scheduler_time", "direction_time",
                                             "construction_time", "local_search_time")},
                   "repair_time": None, "initialization_time": result.initialization_time,
                   "iterations": result.iterations, "candidate_count": result.candidate_count,
                   "reference_calls": result.reference_calls, "certifier_calls": result.certifier_calls,
                   "actual_runtime": result.actual_runtime, "overshoot": result.overshoot,
                   "DEADLOCK": account["baseline_deadlock"], "recovered": account["recovered"],
                   "remaining_DEADLOCK": account["remaining_deadlock"],
                   "numeric_failure_count": account["numeric_failure"],
                   "intermediate_certifier_mismatch_count": None,
                   "cap_hit_telemetry": {field: account[field] for field in CAP_FIELDS},
                   "diagnostics": list(result.diagnostics), "raw_accounting": account}
        metrics["direction_dp_calls"] = direction_count["route_dp_calls"]
        metrics["direction_dp_calls_semantics"] = "observed native per-route fixed-first DP invocations, including initialization; bookkeeping included in native clock"
        for key, value in account.items():
            if key.startswith(("x_pattern_", "y_pattern_", "first_x_")):
                metrics[key] = value
        metrics["decision_family_funnel"] = None
    # Additional independent final check is runner overhead, never backfilled into solver events.
    final_check_started = time.perf_counter()
    certification = None if solution is None or schedule is None else certify_schedule(
        solution, schedule, ScientificConfig(), scope=FORMAL_SCOPE_V2)
    final_check_duration = time.perf_counter() - final_check_started
    final_certified = bool(certification and certification.certified)
    key = run_key(protocol, entry, method, seed)
    catalog = build_legal_pattern_catalog(parents, ScientificConfig(), FORMAL_SCOPE_V2)
    record = {**key, "run_key": key, "run_key_hash": digest(key), "execution_status": "COMPLETED",
              "repository_id": REPOSITORY_ID, "commit_verified": True, "scope_id": FORMAL_SCOPE_V2.scope_id,
              "pattern_catalog_hash": pattern_catalog_hash(parents, ScientificConfig(), FORMAL_SCOPE_V2),
              "workbook": entry["relative_path"], "sheet_name": entry["sheet_name"],
              "tier": entry["tier"], "N": entry["N"], "termination_reason": result.termination_reason,
              "time_to_first_certified": min((t for t, _ in events), default=None),
              "best_events": events, **checkpoint_values(events),
              "final_cmax": None if schedule is None else schedule.cmax,
              "final_schedule_status": None if schedule is None else schedule.status.value,
              "final_certified": final_certified,
              "final_certification_errors": [] if certification is None else list(certification.errors),
              "independent_final_certification_time": final_check_duration,
              "certifier_mismatch_count": int(schedule is not None and schedule.status is ScheduleStatus.FEASIBLE and not final_certified),
              "legal_x_pattern_count": sum(p.kind is SplitKind.X_SPLIT for options in catalog.values() for p in options),
              "legal_y_pattern_count": sum(p.kind is SplitKind.Y_SPLIT for options in catalog.values() for p in options),
              "pattern_counter_semantics": "legacy X/Y count solution presence, including retained patterns; ALNS TARGET family funnel is separate",
              "x_oracle_seed_used": False, "two_opt_star_enabled": False,
              **metrics, **final_structure(solution, directions, schedule)}
    for prefix in ("x", "y"):
        family = None if method != METHODS[0] else metrics["decision_family_funnel"][f"TARGET_{prefix.upper()}"]
        record[f"target_{prefix}_funnel"] = family if family is not None else {
            "generated": None, "constructed": None, "cheap_valid": None, "C2_selected": None,
            "direction_evaluated": None, "reference_evaluated": None, "certified": None,
            "accepted": None, "global_best_update": None,
            "reason": "exact target-family funnel not instrumented by frozen baseline; legacy presence counters in record"}
    if method == METHODS[0]:
        for field in ("first_x_proposal_time", "first_x_c2_time", "first_x_reference_time"):
            record[field] = getattr(stats, field)
    return record


def complete_median(rows, field):
    if len(rows) != 5 or {r["solver_seed"] for r in rows} != set(SEEDS):
        return None
    if not all(r.get("final_certified") and r.get(field) is not None
               and math.isfinite(r[field]) for r in rows):
        return None
    return statistics.median(r[field] for r in rows)


def competition_decision(ratios, *, execution_pass):
    if not execution_pass or len(ratios) != 12 or any(r["R"] is None for r in ratios):
        return {"status": "NOT_EVALUABLE", "passed": False}
    values = [r["R"] for r in ratios]
    tier_medians = {tier: statistics.median(r["R"] for r in ratios if r["tier"] == tier)
                    for tier in ("SMALL", "MEDIUM", "LARGE")}
    median = statistics.median(values)
    count = sum(v <= COMPETITION_GATE["R_threshold"] for v in values)
    passed = (median <= COMPETITION_GATE["median_R_max"] and count >= COMPETITION_GATE["minimum_count"]
              and all(v <= COMPETITION_GATE["tier_median_R_max"] for v in tier_medians.values()))
    return {"status": "COMPETITIVE" if passed else "NEEDS_CANDIDATE_POOL_AUDIT",
            "passed": passed, "median_R": median, "count_R_le_1_10": count,
            "tier_median_R": tier_medians, "predeclared_gate": COMPETITION_GATE}


def summarize(protocol, records, regressions):
    try:
        resume_records = validate_resume({"protocol": protocol, "records": records}, protocol)
        exact_population = len(resume_records) == 180
    except (ValueError, KeyError):
        exact_population = False
    summaries, ratios = [], []
    for entry in protocol["instances"]:
        medians = {}
        for method in METHODS:
            rows = [r for r in records if r["instance_id"] == entry["instance_id"] and r["method_id"] == method]
            median = complete_median(rows, "cmax_at_60")
            medians[method] = median
            values = [] if median is None else [r["cmax_at_60"] for r in rows]
            quartiles = None if not values else statistics.quantiles(values, n=4, method="inclusive")
            summary = {"instance_id": entry["instance_id"], "tier": entry["tier"], "N": entry["N"],
                       "method_id": method, "seeds_present": len(rows),
                       "status": "COMPLETE" if median is not None else "INCOMPLETE_CERTIFICATION",
                       "median_Cmax_5": complete_median(rows, "cmax_at_5"),
                       "median_Cmax_30": complete_median(rows, "cmax_at_30"), "median_Cmax_60": median,
                       "IQR": None if quartiles is None else quartiles[2] - quartiles[0],
                       "mean": statistics.mean(values) if values else None,
                       "std": statistics.stdev(values) if values else None}
            for field in ("time_to_first_certified", "reference_calls", "actual_runtime", "overshoot"):
                summary[f"median_{field}"] = complete_median(rows, field) if median is not None else None
            summaries.append(summary)
        a, h, w = (medians[m] for m in METHODS)
        ratios.append({"instance_id": entry["instance_id"], "tier": entry["tier"],
                       "A": a, "H": h, "W": w,
                       "R": None if any(v is None for v in (a, h, w)) else a / min(h, w)})
    numeric = sum(r.get("numeric_failure_count", 0) for r in records)
    mismatches = sum(r.get("certifier_mismatch_count", 0) + (r.get("intermediate_certifier_mismatch_count") or 0) for r in records)
    certified = sum(bool(r.get("final_certified")) for r in records)
    checkpoint_ok = all(checkpoint_values(r["best_events"]) == {k: r[k] for k in checkpoint_values([])} for r in records)
    catalogs = {}
    for r in records:
        catalogs.setdefault(r["instance_id"], set()).add(r["pattern_catalog_hash"])
    early_cap = any(r["method_id"] == METHODS[0] and r["termination_reason"] == "ITERATION_LIMIT" for r in records)
    integrity = {"numeric_failure": numeric, "scheduler_certifier_mismatch": mismatches,
                 "exact_expected_population": exact_population,
                 "checkpoint_no_backfill": checkpoint_ok,
                 "catalogs_identical": all(len(v) == 1 for v in catalogs.values()),
                 "ALNS_unexpected_iteration_limit": early_cap,
                 "pre_regression_pass": regressions.get("before", {}).get("returncode") == 0,
                 "post_regression_pass": regressions.get("after", {}).get("returncode") == 0}
    passed = (len(records) == certified == 180 and exact_population and numeric == mismatches == 0 and checkpoint_ok
              and integrity["catalogs_identical"] and not early_cap
              and integrity["pre_regression_pass"] and integrity["post_regression_pass"]
              and protocol["protocol_status"] == "FROZEN"
              and protocol["provenance"]["commit_verified"] and not protocol["provenance"]["worktree_dirty"])
    competition = competition_decision(ratios, execution_pass=passed)
    statuses = {"PHASE3Z_EXECUTION_STATUS": "PASS" if passed else "FAIL",
                "VALIDATION_RUNS": f"{len(records)}/180", "CERTIFIED_RUNS": f"{certified}/180",
                "FORMAL_SCOPE_V2_STATUS": "FROZEN_FINAL" if passed else "BLOCKED",
                "DETERMINISTIC_V2_BACKBONE_STATUS": competition["status"],
                "TRADITIONAL_HEURISTIC_TUNING_STATUS": "CLOSED" if passed else "NOT_CLOSED",
                "V2_VALIDATION_STATUS": "CONSUMED" if records else "FROZEN_UNTOUCHED",
                "ID_TEST_STATUS": "SEALED",
                "NEXT_PHASE": "Phase 4-0 — V2 Candidate-Pool Oracle Recall Audit" if passed else "Resolve Phase3-Z execution/provenance blocker only"}
    method_summaries, access = [], []
    for method in METHODS:
        rows = [r for r in records if r["method_id"] == method]
        method_summaries.append({"method_id": method, "runs": len(rows),
            "termination_counts": dict(Counter(r["termination_reason"] for r in rows)),
            **{f"median_{f}": statistics.median(r[f] for r in rows) if rows else None
               for f in ("actual_runtime", "overshoot", "reference_calls")},
            "scheduler_time_total": sum(r["scheduler_time"] for r in rows),
            "cap_hits": {f: sum((r["cap_hit_telemetry"][f] or 0) for r in rows) for f in CAP_FIELDS}})
        for tier in ("SMALL", "MEDIUM", "LARGE"):
            selected = [r for r in rows if r["tier"] == tier]
            reference = sum(r["reference_calls"] for r in selected)
            item = {"method_id": method, "tier": tier, "runs": len(selected)}
            for prefix in ("x", "y"):
                refs = sum(r[f"{prefix}_pattern_reference_evaluated"] for r in selected)
                certs = sum(r[f"{prefix}_pattern_certified"] for r in selected)
                accepts = sum(r[f"{prefix}_pattern_accepted"] for r in selected)
                item[prefix] = {"legal_pattern_count_sum": sum(r[f"legal_{prefix}_pattern_count"] for r in selected),
                                "reference_evaluated": refs, "reference_rate": refs / reference if reference else None,
                                "certified": certs, "accepted": accepts,
                                "acceptance_per_reference": accepts / refs if refs else None,
                                "global_best_updates": sum(r[f"{prefix}_pattern_global_best_updates"] for r in selected),
                                "final_parent_count_sum": sum(r.get(f"final_{prefix}_parent_count", 0) for r in selected)}
            access.append(item)
    efficiency = [{k: r[k] for k in ("instance_id", "tier", "method_id", "solver_seed", "cmax_at_60", "reference_calls", "scheduler_time", "actual_runtime", "overshoot")} for r in records]
    return {"decision": statuses, "integrity": integrity, "competition_decision": competition,
            "per_instance_summaries": summaries, "ratios": ratios,
            "tier_summaries": competition.get("tier_median_R", {}),
            "method_summaries": method_summaries, "access_summaries": access,
            "source_tables": {"A_Cmax60": summaries, "B_R_i": ratios,
                              "C_anytime": [{k: s[k] for k in ("instance_id", "tier", "method_id", "median_Cmax_5", "median_Cmax_30", "median_Cmax_60")} for s in summaries],
                              "D_tier": competition.get("tier_median_R", {}),
                              "E_reference_efficiency": efficiency, "F_X_Y_usage": access}}


def regression():
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run([os.sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                            cwd=ROOT, capture_output=True, text=True)
    return {"returncode": result.returncode, "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(), "output": result.stdout + result.stderr}


def environment():
    return {"Python": platform.python_version(), "executable": os.sys.executable,
            "OS": platform.platform(), "CPU": platform.processor(), "logical_cores": os.cpu_count(),
            "process_mode": "SEQUENTIAL", "thread_env": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "started_at": datetime.now(timezone.utc).isoformat(), "ended_at": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="metadata/protocol only, never open workbooks")
    parser.add_argument("--ppo-root", type=Path, default=Path(os.environ.get("MRTA_PPO_ROOT", "D:/pybullet_test/MRTA_GA/ppo")))
    args = parser.parse_args()
    candidate = build_protocol(require_verified=not args.prepare)
    if PROTOCOL_PATH.exists():
        protocol = read_json(PROTOCOL_PATH)
        if protocol != candidate:
            if protocol.get("protocol_status") == "BLOCKED_BY_PROVENANCE" and not (
                    ARTIFACT_PATH.exists() and read_json(ARTIFACT_PATH).get("records")):
                atomic_json(PROTOCOL_PATH, candidate)
                protocol = candidate
            else:
                raise ValueError("existing frozen protocol differs; refuse overwrite")
    else:
        atomic_json(PROTOCOL_PATH, candidate)
        protocol = candidate
    if args.prepare:
        print(json.dumps({"protocol_status": protocol["protocol_status"],
                          "phase3z_protocol_hash": protocol["phase3z_protocol_hash"],
                          "provenance": protocol["provenance"]}, indent=2))
        return
    verify_protocol(protocol)  # Must precede every workbook-facing path.
    artifact = read_json(ARTIFACT_PATH) if ARTIFACT_PATH.exists() else {
        "protocol": protocol, "records": [], "attempts": [], "regressions": {}, "batches": []}
    if artifact.get("protocol", {}).get("protocol_status") == "BLOCKED_BY_PROVENANCE" and not artifact.get("records"):
        artifact["preparation_history"] = artifact.get("preparation_history", []) + [
            {"protocol": artifact["protocol"], "decision": artifact.get("decision")}]
        artifact["protocol"] = protocol
    completed = validate_resume(artifact, protocol)
    for attempt in artifact.get("attempts", []):
        if attempt.get("execution_status") == "RUNNING":
            attempt.update(execution_status="FAILED_EXECUTION", diagnostic="previous process interrupted before completed record was atomically persisted")
    before = regression()
    artifact.setdefault("regression_attempts", []).append({"stage": "before", **before})
    artifact["regressions"]["before"] = before
    atomic_json(ARTIFACT_PATH, artifact)
    if before["returncode"]:
        raise RuntimeError("pre-run full regression failed; no validation opened")
    verify_protocol(protocol)
    batch = {**environment(), **protocol["provenance"]}
    artifact["batches"].append(batch)
    roles, _, _ = frozen_metadata()
    entries = {e["instance_id"]: e for e in protocol["instances"]}
    parents_cache = {}
    for order in protocol["run_order"]:
        entry = entries[order["instance_id"]]
        method, seed = order["method_id"], order["solver_seed"]
        key = run_key(protocol, entry, method, seed)
        if digest(key) in completed:
            continue
        verify_protocol(protocol)
        attempt = {"run_key": key, "started_at": datetime.now(timezone.utc).isoformat(),
                   "execution_status": "RUNNING"}
        artifact["attempts"].append(attempt)
        atomic_json(ARTIFACT_PATH, artifact)
        try:
            if entry["instance_id"] not in parents_cache:
                parents_cache[entry["instance_id"]] = load_validation_parents(roles, entry, args.ppo_root)
            record = run_method(protocol, entry, method, seed, parents_cache[entry["instance_id"]])
        except BaseException as error:
            attempt.update(execution_status="FAILED_EXECUTION", diagnostic=str(error),
                           traceback_summary=traceback.format_exc(), ended_at=datetime.now(timezone.utc).isoformat())
            artifact["execution_blocker"] = attempt
            atomic_json(ARTIFACT_PATH, artifact)
            raise
        attempt.update(execution_status="COMPLETED", ended_at=datetime.now(timezone.utc).isoformat())
        artifact["records"].append(record)
        completed[digest(key)] = record
        artifact.update(summarize(protocol, artifact["records"], artifact["regressions"]))
        atomic_json(ARTIFACT_PATH, artifact)
        print(f"Z {len(completed)}/180 {entry['tier']} {method} seed={seed} certified={record['final_certified']} Cmax60={record['cmax_at_60']}", flush=True)
        if (not record["final_certified"] or record["numeric_failure_count"] or record["certifier_mismatch_count"]
                or record.get("intermediate_certifier_mismatch_count")
                or (method == METHODS[0] and record["termination_reason"] == "ITERATION_LIMIT")):
            raise RuntimeError("certification/numeric failure; stop without tuning")
    after = regression()
    artifact["regressions"]["after"] = after
    artifact["regression_attempts"].append({"stage": "after", **after})
    verify_protocol(protocol)
    batch["ended_at"] = datetime.now(timezone.utc).isoformat()
    artifact.update(summarize(protocol, artifact["records"], artifact["regressions"]))
    atomic_json(ARTIFACT_PATH, artifact)
    print(json.dumps(artifact["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
