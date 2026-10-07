"""Development-only Phase3-ZR measurements; historical validation stays read-only.

Observers wrap existing calls and return the very same objects. Level 1 records
coarse timings only; level 2 enables the existing SchedulerProfile on replays.
Run from this checkout with the specified interpreter using -B -m scripts....
"""
from __future__ import annotations

import argparse
import cProfile
from copy import deepcopy
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from functools import wraps
import hashlib
import json
import math
import os
import pstats
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
# The shared venv has an editable install of the outer copy. Select this exact
# checkout before importing scientific modules; workbook storage is independent.
sys.path.insert(0, str(ROOT / "src"))
DATA_ROOT = ROOT.parents[2] / "ppo" if ROOT.parent.name == "workspaces" else ROOT.parent / "ppo"

from scripts import run_phase3z_v2_validation as historical
from scripts.run_phase3y_v2_core import _geometry_descriptor, _valid_entries
from mrta_baselines.common import CommonBaselineEvaluator
from mrta_baselines.hga import run_adapted_hga
from mrta_baselines.wag_vns import run_adapted_wag_vns
from mrta_data.phase3_split import assert_v2_solver_access_allowed
from mrta_data.ppo_instances import load_ppo_platform_instance, instance_geometry_hash, to_parent_welds
from mrta_reference import scheduler
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import build_legal_pattern_catalog
from mrta_reference.model import ParentWeld, Route, SplitPattern, SplitKind, Rail, ScientificConfig, ScheduleStatus
from mrta_reference.provenance import resolve_source_provenance
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize
from mrta_search.pipeline import run_sa_oi_alns_v2
from mrta_search.stats import SearchStats

PROTOCOL = ROOT / "data/manifests/PHASE3ZR_RUNTIME_PROTOCOL_V1.json"
RUNTIME_SET = ROOT / "data/manifests/PPO_PHASE3ZR_RUNTIME_SET_V1.json"
AUDIT = ROOT / "data/development/phase3zr_reference_runtime_audit_v1.json"
SLOW = ROOT / "data/development/phase3zr_slow_reference_calls_v1.json"
HGA = ROOT / "data/development/phase3zr_hga_incumbent_robustness_v1.json"
REPORT = ROOT / "docs/PHASE3ZR_REFERENCE_RUNTIME_AND_INCUMBENT_ROBUSTNESS_20261006.md"
ROLE = "V2_MODEL_DEVELOPMENT_CONSUMED"
METHODS = historical.METHODS
SEEDS = (20261021, 20261022)
HGA_SEEDS = (*SEEDS, 20261023, 20261024, 20261025)
FIELDS = ("N", "total_weld_length", "bbox_coverage", "x_coverage", "y_coverage",
          "upper_lower_imbalance", "left_right_imbalance", "x_splittable_process_share",
          "max_x_span", "blocking_proxy_total", "x_split_pattern_count", "y_split_pattern_count")
THRESHOLDS = {"p95": 8.0, "p99": 30.0, "max": 60.0, "run_max": 120.0}
digest = historical.digest
STORAGE_FORMAT = "phase3zr_replay_tables_v1"



def pack_replays(artifact):
    """Lossless in-file deduplication; integer IDs are table offsets, not hashes."""
    if "storage_format" in artifact or "replay_tables" in artifact:
        raise ValueError("pack_replays requires an expanded artifact")
    names = ("parents", "patterns", "pattern_sets", "block_ids", "routes",
             "directions", "metadata", "replays")
    tables = {name: [] for name in names}
    indexes = {name: {} for name in names}

    def intern(name, value):
        key = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
        if key not in indexes[name]:
            indexes[name][key] = len(tables[name])
            tables[name].append(deepcopy(value))
        return indexes[name][key]

    def replay_id(value):
        patterns = [intern("patterns", p) for p in value["patterns"]]
        routes = [intern("routes", {**route, "block_ids":
                  [intern("block_ids", b) for b in route["block_ids"]]})
                  for route in value["routes"]]
        return intern("replays", {
            "parents": intern("parents", value["parents"]),
            "patterns": intern("pattern_sets", patterns),
            "routes": routes,
            "directions": intern("directions", value["directions"]),
            "metadata": intern("metadata", {k: v for k, v in value.items()
                if k not in {"parents", "patterns", "routes", "directions"}})})

    def walk(value):
        if isinstance(value, dict):
            if "replay_ref" in value:
                raise ValueError("expanded artifact contains reserved replay_ref")
            return {("replay_ref" if k == "replay" else k):
                    (replay_id(value[k]) if k == "replay" else walk(value[k]))
                    for k in sorted(value)}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    result = walk(artifact)
    result["storage_format"] = STORAGE_FORMAT
    result["replay_tables"] = tables
    return result


def unpack_replays(artifact):
    """Read old inline JSON or restore independent per-call replay payloads."""
    if "storage_format" not in artifact and "replay_tables" not in artifact:
        return artifact
    if artifact.get("storage_format") != STORAGE_FORMAT:
        raise ValueError("unsupported replay storage format")
    tables = artifact["replay_tables"]

    def lookup(name, index):
        rows = tables.get(name)
        if not isinstance(rows, list) or type(index) is not int or not 0 <= index < len(rows):
            raise ValueError(f"invalid {name} reference: {index!r}")
        return rows[index]

    def expand(index):
        row = lookup("replays", index)
        routes = [{**route, "block_ids": [lookup("block_ids", b) for b in route["block_ids"]]}
                  for route in (lookup("routes", i) for i in row["routes"])]
        return deepcopy({**lookup("metadata", row["metadata"]),
            "parents": lookup("parents", row["parents"]),
            "patterns": [lookup("patterns", i) for i in lookup("pattern_sets", row["patterns"])],
            "routes": routes, "directions": lookup("directions", row["directions"])})

    def walk(value):
        if isinstance(value, dict):
            return {("replay" if k == "replay_ref" else k):
                    (expand(v) if k == "replay_ref" else walk(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk({k: v for k, v in artifact.items() if k not in {"storage_format", "replay_tables"}})


def read(path):
    result = historical.read_json(path)
    return unpack_replays(result) if Path(path).resolve() == AUDIT.resolve() else result


def compact_storage():
    """Rewrite only AUDIT, verifying the whole expanded artifact before/after."""
    original = read(AUDIT)
    before = AUDIT.stat().st_size
    write(AUDIT, original)
    if read(AUDIT) != original:
        raise ValueError("stored audit differs after replay expansion")
    stored = historical.read_json(AUDIT)
    after = AUDIT.stat().st_size
    print(json.dumps({"bytes_before": before, "bytes_after": after,
        "reduction_percent": 100.0*(before-after)/before,
        "all_fields_equal_after_expansion": True,
        "table_counts": {k: len(v) for k, v in stored["replay_tables"].items()}}, indent=2), flush=True)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(path, payload):
    # The sole persistence boundary is an explicit list of the approved artifacts.
    if Path(path).resolve() not in {PROTOCOL, RUNTIME_SET, AUDIT, SLOW, HGA}:
        raise ValueError("ZR cannot write historical or arbitrary paths")
    # Candidate payloads are large. Encode once using compact JSON, outside the
    # native solver clock; keep the same fsync + same-directory atomic replace.
    path = Path(path)
    prepared = historical.json_ready(payload)
    stored = pack_replays(prepared) if path.resolve() == AUDIT.resolve() else prepared
    if path.resolve() == AUDIT.resolve() and unpack_replays(stored) != prepared:
        raise ValueError("audit storage conversion changed content")
    encoded = json.dumps(stored, sort_keys=True,
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def outcome(*, final_certified=False, numeric=0, exception=False):
    if exception:
        return "EXECUTION_FAILURE"
    if numeric:
        return "NUMERIC_FAILURE"
    return "CERTIFIED_INCUMBENT" if final_certified else "NO_CERTIFIED_INCUMBENT"


def load_parents(entry, *, forensic=False, loader=None):
    roles = read(historical.ROLES_PATH)
    if forensic:
        z11 = read(historical.PROTOCOL_PATH)["instances"][10]
        if entry != z11:
            raise ValueError("dedicated forensic entry permits frozen Z11 only")
        assert_v2_solver_access_allowed(roles, [entry["relative_path"]], allowed_roles=("V2_VALIDATION",))
    else:
        assert_v2_solver_access_allowed(roles, [entry["relative_path"]], allowed_roles=(ROLE,))
    instance = (loader or load_ppo_platform_instance)(DATA_ROOT / entry["relative_path"], entry["sheet_name"],
                ppo_root=DATA_ROOT, instance_id=entry["instance_id"])
    if instance.raw_file_sha256 != entry["workbook_sha256"] or instance_geometry_hash(instance) != entry["instance_geometry_hash"]:
        raise ValueError("instance differs from selected manifest")
    parents = to_parent_welds(instance)
    if len(parents) != entry["N"]:
        raise ValueError("N differs")
    return parents


def provenance():
    historical.verify_imported_source()
    return asdict(resolve_source_provenance(ROOT))


def prepare():
    if PROTOCOL.exists() or RUNTIME_SET.exists():
        raise ValueError("already prepared; existing preregistration is never overwritten")
    roles = read(historical.ROLES_PATH)
    by_path = {e["relative_path"]: e["new_v2_role"] for e in roles["workbooks"]}
    entries = [e for e in _valid_entries(read(ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json"))
               if by_path.get(e["relative_path"]) == ROLE]
    selected, used = [], set()
    # Select LARGE first to leave eight distinct workbooks for the robustness audit.
    for tier, low, high, target, count, pool_size in (("LARGE", 75, 90, 85, 8, 24), ("MEDIUM", 50, 60, 55, 4, 12)):
        grouped = {}
        for e in entries:
            if e["relative_path"] not in used:
                grouped.setdefault(e["relative_path"], []).append(e)
        key = lambda e: (0 if low <= e["actual_weld_count"] <= high else 1,
                         abs(e["actual_weld_count"] - target), e["instance_geometry_hash"])
        candidates = sorted((min(rows, key=key) for rows in grouped.values()), key=key)[:pool_size]
        if len(candidates) < count:
            raise ValueError("insufficient distinct development workbooks")
        described = []
        for e in candidates:
            assert_v2_solver_access_allowed(roles, [e["relative_path"]], allowed_roles=(ROLE,))
            d = _geometry_descriptor(e, DATA_ROOT, ScientificConfig())
            temp_entry = {**e, "workbook_sha256": e["raw_file_sha256"], "N": e["actual_weld_count"]}
            catalog = build_legal_pattern_catalog(load_parents(temp_entry), ScientificConfig(), FORMAL_SCOPE_V2)
            d["y_split_pattern_count"] = sum(p.kind is SplitKind.Y_SPLIT for opts in catalog.values() for p in opts)
            described.append((e, d))
        mins = {f: min(float(d[f]) for _, d in described) for f in FIELDS}
        maxs = {f: max(float(d[f]) for _, d in described) for f in FIELDS}
        vectors = {e["instance_id"]: tuple(0.0 if maxs[f] == mins[f] else (float(d[f])-mins[f])/(maxs[f]-mins[f]) for f in FIELDS) for e, d in described}
        chosen = [min(described, key=lambda item: key(item[0]))]
        while len(chosen) < count:
            rest = [item for item in described if item not in chosen]
            chosen.append(min(rest, key=lambda item: (
                -min(math.dist(vectors[item[0]["instance_id"]], vectors[old[0]["instance_id"]]) for old in chosen),
                item[0]["instance_geometry_hash"])))
        for e, d in chosen:
            used.add(e["relative_path"])
            selected.append({"instance_id": e["instance_id"], "relative_path": e["relative_path"],
                "sheet_name": e["sheet_name"], "instance_geometry_hash": e["instance_geometry_hash"],
                "workbook_sha256": e["raw_file_sha256"], "N": e["actual_weld_count"], "tier": tier,
                "new_v2_role": ROLE, "geometry_descriptor": d, "normalized_descriptor": vectors[e["instance_id"]]})
    runtime = {"runtime_set_id": "PPO_PHASE3ZR_RUNTIME_SET_V1", "created_at": now(),
        "selection_policy": {"workbook_representative": "preferred range then nearest target N; geometry hash tie",
            "candidate_pool": "24 closest distinct workbooks LARGE; remaining 12 closest MEDIUM",
            "first": "closest preferred-range N then geometry hash", "next": "deterministic farthest-first; geometry hash tie",
            "descriptor_fields": FIELDS, "solver_results_used": False}, "instances": selected}
    runtime["runtime_set_hash"] = digest(runtime)
    old = read(historical.ARTIFACT_PATH)
    protocol = {"protocol_id": "PHASE3ZR_RUNTIME_PROTOCOL_V1", "created_at": now(),
        "scope_id": FORMAL_SCOPE_V2.scope_id, "scope_hash": FORMAL_SCOPE_V2.scope_hash,
        "runtime_set_hash": runtime["runtime_set_hash"], "allowed_role": ROLE,
        "stage_a_seeds": SEEDS, "hga_seeds": HGA_SEEDS, "requested_budget": 60.0,
        "methods": read(historical.PROTOCOL_PATH)["methods"], "scientific_config": asdict(ScientificConfig()),
        "runtime_thresholds": THRESHOLDS, "hga_classes": {"ROBUST_95": ">=38/40", "INTERMITTENT_NO_CERTIFIED": "32..37/40", "SYSTEMIC_NO_CERTIFIED": "<32/40"},
        "execution": "sequential; native initialization in clock; no injection, retries, timeout or changed policy",
        "level_1": "coarse reference/preparation/baseline/recovery/packaging timing; candidate serialization outside native clock",
        "level_2": "existing counters plus per-rollout timing; diagnostic only, excluded from thresholds",
        "slow_set_policy": "top20 unique geometry/solution/directions; add5 FEASIBLE,5 recovered FEASIBLE,5 remaining DEADLOCK if available",
        "stage_a_count": 72, "hga_count": 40, "provenance_before_instrumentation": provenance(),
        "workbook_storage": str(DATA_ROOT),
        "remote_main_verification": "network unavailable; supplied 94ea6c1 not verified",
        "historical_execution": "FAIL_179_OF_180_CERTIFIED", "historical_artifact_read_only": True,
        "historical_run_count": len(old["records"]), "regression_before": "314 passed in 61.05s",
        "future_outcome_policy": "EXECUTION_FAILURE for exception; NUMERIC_FAILURE for numeric; otherwise certified or no certified incumbent; preserve old termination_reason",
        "forensic_policy": "Z11 only after Stage A/B root cause and HGA audit; separate ZR storage; never replace history"}
    protocol["protocol_hash"] = digest(protocol)
    write(PROTOCOL, protocol)
    write(RUNTIME_SET, runtime)
    # Read-only is enforced by write()'s exact artifact allowlist. The original
    # evidence is only opened for reads; no attributes or content are rewritten.
    print(json.dumps({"prepared": True, "instances": [(e["tier"], e["N"], e["instance_id"]) for e in selected]}, indent=2), flush=True)


def payload(solution, config, directions):
    return {"parents": [asdict(p) for p in solution.parents], "patterns": [asdict(p) for p in solution.patterns],
            "routes": [asdict(r) for r in solution.routes], "revision": solution.revision,
            "directions": directions, "scientific_config": asdict(config), "scope_id": FORMAL_SCOPE_V2.scope_id,
            "scope_hash": FORMAL_SCOPE_V2.scope_hash}


def reconstruct(data):
    if data["scope_hash"] != FORMAL_SCOPE_V2.scope_hash:
        raise ValueError("replay scope differs")
    config = ScientificConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["scientific_config"].items()})
    parents = tuple(ParentWeld(p["parent_id"], tuple(p["start"]), tuple(p["end"])) for p in data["parents"])
    patterns = tuple(SplitPattern(p["parent_id"], SplitKind(p["kind"]), p["t"], p["point_id"], p["mandatory"], None if p["rail"] is None else Rail(p["rail"])) for p in data["patterns"])
    routes = tuple(Route(r["robot_id"], tuple(r["block_ids"])) for r in data["routes"])
    solution = canonicalize(parents, patterns, routes, config, revision=data["revision"], scope=FORMAL_SCOPE_V2)
    return solution, config, {r: tuple(v) for r, v in enumerate(data["directions"])}


class ReferenceTrace:
    """Single-threaded scoped observer; failures are explicit execution failures.

    No scheduling call is repeated to manufacture evidence. Existing certifier
    results are observed. The context always restores every patched callable.
    """
    def __init__(self, method, level=1, sink=None):
        if level not in (0, 1, 2):
            raise ValueError("telemetry level must be 0, 1 or 2")
        self.method, self.level, self.sink = method, level, sink
        self.rows, self.errors, self.patches = [], [], []
        self.current, self.source, self.in_recovery = None, "FORMAL_REFERENCE", False
        self.started = time.perf_counter()

    def patch(self, obj, name, replacement):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def __enter__(self):
        if self.level == 0:
            return self
        original = scheduler.FormalReferenceEvaluator.__call__
        @wraps(original)
        def reference(evaluator, solution, config, *, orientations=None):
            start = time.perf_counter()
            row = {"reference_call_index": len(self.rows)+1, "method_id": self.method,
                   "candidate_source": self.source, "scheduler_start_time": start-self.started,
                   "baseline_dispatch_time": 0.0, "recovery_time": 0.0, "preparation_time": 0.0,
                   "packaging_time": 0.0, "rollouts": [], "certified": None}
            if self.level == 2:
                row.update(rollouts_attempted=0, rollouts_completed=0,
                           selected_rollout_index=None, selected_plan=[])
            previous = self.current
            self.current = row
            profile = scheduler.SchedulerProfile() if self.level == 2 else None
            row["_absolute_start"] = start
            try:
                if profile is None:
                    result = original(evaluator, solution, config, orientations=orientations)
                else:
                    if evaluator.deadlock_observer is not None:
                        raise ValueError("deep replay excludes extra deadlock callback evaluations")
                    result = scheduler.reference_schedule_formal(solution, config, scope=evaluator.scope,
                                orientations=orientations, profile=profile)
            except Exception:
                row["exception"] = traceback.format_exc()
                self.errors.append(row["exception"])
                raise
            finally:
                end = time.perf_counter()
                row["scheduler_end_time"] = end-self.started
                row["scheduler_duration"] = end-start
                if "_baseline_start" not in row:
                    row["preparation_time"] = end-start
                row["packaging_time"] = max(0.0, end-start-row["preparation_time"]-row["baseline_dispatch_time"]-row["recovery_time"])
                self.current = previous
            row.update({"schedule_status": result.status.value, "schedule_source": result.source,
                "baseline_deadlock": result.baseline_deadlock, "recovered": result.baseline_deadlock and result.feasible,
                "remaining_deadlock": result.status is ScheduleStatus.DEADLOCK, "Cmax": result.cmax if result.feasible else None,
                **{k: getattr(result, k) for k in ("expanded_states", "state_budget", "recovery_rollouts", "rollout_budget", "max_discrepancies_used", "frontier_exhausted", "branch_points_considered")},
                "_solution": solution, "_config": config, "_schedule": result,
                "profile": None if profile is None else profile.as_dict()})
            self.rows.append(row)
            if self.sink is not None:
                self.sink(row)  # explicit fail-closed; never silently omit measurement
            return result
        self.patch(scheduler.FormalReferenceEvaluator, "__call__", reference)

        original_evaluate = CommonBaselineEvaluator.evaluate
        @wraps(original_evaluate)
        def evaluate(evaluator, solution, *, source):
            previous, self.source = self.source, source
            calls_before = len(self.rows)
            try:
                result = original_evaluate(evaluator, solution, source=source)
                if len(self.rows) > calls_before and self.rows[-1]["_solution"] is solution:
                    certification = result.certification
                    self.rows[-1]["certified"] = bool(certification and certification.certified)
                    self.rows[-1]["certification_errors"] = [] if certification is None else list(certification.errors)
                return result
            finally:
                self.source = previous
        self.patch(CommonBaselineEvaluator, "evaluate", evaluate)

        original_dispatch = scheduler._prepared_dispatch_outcome
        @wraps(original_dispatch)
        def dispatch(*args, **kwargs):
            row = self.current
            if row is None:
                return original_dispatch(*args, **kwargs)
            started = time.perf_counter()
            recovering = self.in_recovery
            if not recovering:
                row["_baseline_start"] = started
                row["preparation_time"] = started-row["_absolute_start"]
            result = original_dispatch(*args, **kwargs)
            duration = time.perf_counter()-started
            if not recovering:
                row["baseline_dispatch_time"] += duration
            elif self.level == 2:
                row["rollouts"].append({"index": len(row["rollouts"])+1, "duration": duration,
                    "status": result.result.status.value, "Cmax": result.result.cmax,
                    "forced_decisions": [asdict(d) for d in result.trace.forced_decisions],
                    "initial_depth": result.trace.initial_depth, "terminal_depth": result.trace.terminal_depth,
                    "branch_points": len(result.trace.branch_points),
                    "actual_discrepancies": sum(b.chosen_choice_rank != 0 for b in result.trace.branch_points),
                    "_result": result.result})
            return result
        self.patch(scheduler, "_prepared_dispatch_outcome", dispatch)

        original_recovery = scheduler._limited_discrepancy_dispatch_recovery_optimized
        @wraps(original_recovery)
        def recovery(*args, **kwargs):
            started = time.perf_counter()
            previous, self.in_recovery = self.in_recovery, True
            try:
                return original_recovery(*args, **kwargs)
            finally:
                self.in_recovery = previous
                if self.current is not None:
                    self.current["recovery_time"] += time.perf_counter()-started
        self.patch(scheduler, "_limited_discrepancy_dispatch_recovery_optimized", recovery)

        # Attach the actual selected plan without changing frontier order or selection.
        original_summary = scheduler.limited_discrepancy_recovery
        @wraps(original_summary)
        def summary(*args, **kwargs):
            result = original_summary(*args, **kwargs)
            if self.current is not None and self.level == 2:
                self.current["selected_plan"] = [asdict(d) for d in result.best_plan]
                self.current["rollouts_attempted"] = result.recovery_rollouts
                self.current["rollouts_completed"] = len(self.current["rollouts"])
                self.current["selected_rollout_index"] = next((r["index"] for r in self.current["rollouts"] if r["_result"] is result.result), None)
            return result
        self.patch(scheduler, "limited_discrepancy_recovery", summary)

        # Formal initialization validates certifier identity against a default
        # argument bound at import time. Observe the existing post-certification
        # accounting instead of wrapping that protected certifier callable.
        original_record = SearchStats.record_reference
        @wraps(original_record)
        def recorded(stats, status, duration, **kwargs):
            result = original_record(stats, status, duration, **kwargs)
            schedule = kwargs.get("schedule")
            if self.rows:
                row = self.rows[-1]
                if schedule is row["_schedule"] or (schedule is not None and schedule.directions == row["_schedule"].directions):
                    row["certified"] = status is ScheduleStatus.FEASIBLE
                    row["candidate_source"] = "ALNS_INITIALIZATION" if kwargs["initialization"] else "ALNS_REFERENCE"
                    row["certification_errors"] = list(schedule.diagnostics) if status is ScheduleStatus.NUMERIC_FAILURE else []
            return result
        self.patch(SearchStats, "record_reference", recorded)
        return self

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)

    def serialize(self, entry):
        rows = []
        for raw in self.rows:
            row = {k: v for k, v in raw.items() if not k.startswith("_")}
            solution, config, schedule = raw["_solution"], raw["_config"], raw["_schedule"]
            if schedule.scope_hash != FORMAL_SCOPE_V2.scope_hash:
                raise ValueError("ZR replay artifacts require the frozen V2 scope")
            directions = schedule.directions
            row.update({"instance_id": entry["instance_id"], "instance_geometry_hash": entry["instance_geometry_hash"],
                "N": len(solution.parents), "tier": entry["tier"], "block_count": sum(len(r.block_ids) for r in solution.routes),
                "solution_canonical_hash": solution.canonical_hash, "direction_hash": digest(directions),
                "robot_route_lengths": [len(r.block_ids) for r in solution.routes],
                "WHOLE_count": sum(p.kind is SplitKind.WHOLE for p in solution.patterns),
                "X_count": sum(p.kind is SplitKind.X_SPLIT for p in solution.patterns),
                "Y_count": sum(p.kind is SplitKind.Y_SPLIT for p in solution.patterns),
                "replay": payload(solution, config, directions), "level": self.level})
            # Ordinary compound primary key from the requested identities; no
            # extra hash/contract is needed to identify a unique reference call.
            row["call_identity"] = ":".join((entry["instance_geometry_hash"], row["solution_canonical_hash"], row["direction_hash"]))
            row["rollouts"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in row["rollouts"]]
            if row["certified"] is None and not schedule.feasible:
                row["certified"] = False
            rows.append(historical.json_ready(row))
        return rows


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, **{k: None for k in ("p50", "p90", "p95", "p99", "max")}}
    def quantile(p):
        index = (len(ordered)-1)*p
        low = math.floor(index)
        return ordered[low]+(ordered[math.ceil(index)]-ordered[low])*(index-low)
    return {"count": len(ordered), **{f"p{int(p*100)}": quantile(p) for p in (.5, .9, .95, .99)}, "max": ordered[-1]}


def aggregate(records):
    calls = [c for r in records for c in r.get("reference_trace", [])]
    groups = {"ALL": calls}
    for method in METHODS:
        groups[method] = [c for c in calls if c["method_id"] == method]
    for tier in ("SMALL", "MEDIUM", "LARGE"):
        groups[tier] = [c for c in calls if c["tier"] == tier]
        for method in METHODS:
            groups[f"{method}/{tier}"] = [c for c in calls if c["tier"] == tier and c["method_id"] == method]
    groups.update({"FEASIBLE": [c for c in calls if c["schedule_status"] == "FEASIBLE"],
        "BASELINE_FEASIBLE": [c for c in calls if c["schedule_status"] == "FEASIBLE" and not c["baseline_deadlock"]],
        "RECOVERED_FEASIBLE": [c for c in calls if c["recovered"]],
        "REMAINING_DEADLOCK": [c for c in calls if c["remaining_deadlock"]],
        "X_PRESENT": [c for c in calls if c["X_count"]], "Y_PRESENT": [c for c in calls if c["Y_count"]],
        "WHOLE_ONLY": [c for c in calls if not c["X_count"] and not c["Y_count"]]})
    stats = {k: distribution([c["scheduler_duration"] for c in group]) for k, group in groups.items()}
    return {"runs": len(records), "outcomes": dict(Counter(r["solver_outcome_class"] for r in records)),
        "reference_distribution": stats, "run_runtime": distribution([r["actual_runtime"] for r in records]),
        "overshoot": distribution([r["overshoot"] for r in records]),
        "time_to_first_certified": distribution([r["time_to_first_certified"] for r in records if r["time_to_first_certified"] is not None]),
        "baseline_deadlock": sum(c["baseline_deadlock"] for c in calls), "recovered": sum(c["recovered"] for c in calls),
        "remaining_deadlock": sum(c["remaining_deadlock"] for c in calls),
        "component_totals": {f: sum(c[f] for c in calls) for f in ("preparation_time", "baseline_dispatch_time", "recovery_time", "packaging_time")},
        "numeric_failure": sum(r["numeric_failure_count"] for r in records),
        "certifier_mismatch": sum(r["certifier_mismatch_count"] for r in records),
        "execution_failure": sum(r["solver_outcome_class"] == "EXECUTION_FAILURE" for r in records)}


def run_one(entry, method, seed, *, forensic=False):
    if forensic:
        if method != METHODS[1] or seed != 20261015:
            raise PermissionError("external forensic permits frozen Z11/HGA/20261015 only")
        if not SLOW.exists() or not HGA.exists() or read(SLOW)["status"] != "COMPLETE" or read(HGA)["status"] != "COMPLETE":
            raise PermissionError("development diagnosis and HGA audit must precede external access")
    parents = load_parents(entry, forensic=forensic)
    configs = historical.method_configs()
    source = provenance()
    with ReferenceTrace(method) as trace:
        try:
            if method == METHODS[0]:
                result = run_sa_oi_alns_v2(parents, ScientificConfig(), configs[method], seed=seed,
                            source_provenance=resolve_source_provenance(ROOT), formal_result=False)
                solution, schedule = result.best_solution, result.best_schedule
                events = result.stats.best_events
                runtime, overshoot = result.runtime, result.stats.overshoot
                numeric = sum(r["status"] == "NUMERIC_FAILURE" for r in result.stats.reference_records)
                numeric += int(result.status.value == "NUMERIC_FAILURE")
                mismatch = sum(any("failed certification" in str(d) for d in r["diagnostics"]) for r in result.stats.reference_records)
                initial_certified, initial_refs = None, result.stats.init_reference_calls if hasattr(result.stats, "init_reference_calls") else None
                raw = historical.json_ready(result.stats)
            else:
                runner = run_adapted_hga if method == METHODS[1] else run_adapted_wag_vns
                result = runner(parents, ScientificConfig(), configs[method], seed=seed, time_limit=60.0,
                            checkpoints=historical.CHECKPOINTS, scope=FORMAL_SCOPE_V2, method_id=method,
                            **({} if method == METHODS[1] else {"pattern_access_policy": True}))
                solution, schedule = result.solution, result.schedule
                events = [(e.elapsed, e.cmax) for e in result.best_events]
                runtime, overshoot = result.actual_runtime, result.overshoot
                numeric = result.accounting["numeric_failure"]
                mismatch = sum(bool(c.get("certification_errors")) for c in trace.rows)
                initial = result.initial_candidate
                initial_certified = bool(initial is not None and initial.certification is not None and initial.certification.certified)
                initial_refs = result.initial_reference_calls
                raw = historical.json_ready(result.accounting)
            certification = None if solution is None or schedule is None else certify_schedule(solution, schedule, ScientificConfig(), scope=FORMAL_SCOPE_V2)
            final = bool(certification and certification.certified)
            mismatch += int(schedule is not None and schedule.feasible and not final)
            record = {"execution_status": "COMPLETED", "termination_reason": result.termination_reason,
                "final_certified": final, "final_Cmax": None if schedule is None else schedule.cmax,
                "solver_outcome_class": outcome(final_certified=final, numeric=numeric),
                "numeric_failure_count": numeric, "certifier_mismatch_count": mismatch,
                "actual_runtime": runtime, "overshoot": overshoot,
                "time_to_first_certified": min((t for t, _ in events), default=None),
                "initial_candidate_certified": initial_certified, "initial_reference_calls": initial_refs,
                "raw_accounting": raw}
        except Exception:
            record = {"execution_status": "FAILED", "solver_outcome_class": "EXECUTION_FAILURE",
                "exception": traceback.format_exc(), "termination_reason": None, "final_certified": False,
                "numeric_failure_count": 0, "certifier_mismatch_count": 0, "time_to_first_certified": None,
                "actual_runtime": time.perf_counter()-trace.started, "overshoot": max(0.0,time.perf_counter()-trace.started-60.0)}
    record.update({"instance_id": entry["instance_id"], "tier": entry["tier"], "N": entry["N"],
        "method_id": method, "solver_seed": seed, "requested_budget": 60.0,
        "provenance": source, "reference_implementation_id": "PRE_ZR_B32_43a004c_OBSERVER_ONLY", "observer_version": "POST_CERTIFICATION_ACCOUNTING_V2",
        "reference_trace": trace.serialize(entry), "scope_hash": FORMAL_SCOPE_V2.scope_hash,
        "runtime_set_hash": read(RUNTIME_SET)["runtime_set_hash"],
        "method_config": historical.json_ready(configs[method]), "completed_at": now(),
        "data_role": "EXTERNAL_FROZEN_STRESS_CASE" if forensic else ROLE})
    record["reference_call_count"] = len(trace.rows)
    return record


def require_prepared():
    p, s = read(PROTOCOL), read(RUNTIME_SET)
    if p["runtime_set_hash"] != s["runtime_set_hash"] or digest({k:v for k,v in s.items() if k != "runtime_set_hash"}) != s["runtime_set_hash"]:
        raise ValueError("runtime set changed")
    if digest({k:v for k,v in p.items() if k != "protocol_hash"}) != p["protocol_hash"]:
        raise ValueError("protocol changed")
    return p, s


def stage_a():
    p, s = require_prepared()
    artifact = read(AUDIT) if AUDIT.exists() else {"protocol_hash": p["protocol_hash"], "status": "RUNNING", "records": [], "started_at": now()}
    if artifact["protocol_hash"] != p["protocol_hash"]:
        raise ValueError("resume protocol differs")
    completed = {(r["instance_id"], r["method_id"], r["solver_seed"]) for r in artifact["records"]}
    if len(completed) != len(artifact["records"]):
        raise ValueError("duplicate run records")
    for i, entry in enumerate(s["instances"]):
        for j, seed in enumerate(SEEDS):
            offset = (i+j)%3
            for method in METHODS[offset:]+METHODS[:offset]:
                key = (entry["instance_id"], method, seed)
                if key in completed:
                    continue
                print(f"START {len(artifact['records'])+1}/72 {entry['N']} {method} {seed}", flush=True)
                row = run_one(entry, method, seed)
                artifact["records"].append(row)
                artifact["summary"] = aggregate(artifact["records"])
                write(AUDIT, artifact)
                completed.add(key)
                print(f"DONE {len(artifact['records'])}/72 {row['solver_outcome_class']} runtime={row['actual_runtime']:.3f} calls={row['reference_call_count']}", flush=True)
    artifact["status"], artifact["finished_at"] = "COMPLETE", now()
    write(AUDIT, artifact)


def repair_startup():
    """Preserve unreached ALNS startup attempts; never retry a solver outcome.

    This is specific to the observer's certifier-identity error, which occurred
    before direction/reference evaluation. Completed HGA/WAG runs are retained.
    It cannot move no-certified/numeric outcomes or any evaluated candidate.
    """
    artifact = read(AUDIT)
    if artifact["status"] != "COMPLETE":
        raise ValueError("original process must finish before repairing startup instrumentation")
    retained, startup = [], []
    for row in artifact["records"]:
        if (row["method_id"] == METHODS[0] and row["solver_outcome_class"] == "EXECUTION_FAILURE"
            and row["reference_call_count"] == 0
            and "formal initialization requires the common certifier" in row.get("exception", "")):
            startup.append(row)
        else:
            row["observer_version"] = row.get("observer_version", "CERTIFIER_WRAPPER_V1")
            for call in row["reference_trace"]:
                call["call_identity"] = ":".join((call["instance_geometry_hash"], call["solution_canonical_hash"], call["direction_hash"]))
            retained.append(row)
    if not startup:
        raise ValueError("no unreached observer startup errors to repair")
    artifact.setdefault("observer_startup_attempts", []).extend(startup)
    artifact["records"] = retained
    artifact["status"] = "RUNNING"
    artifact["startup_repair"] = {"at": now(), "reason": "protected certifier identity restored; use post-certification accounting",
        "unreached_attempts_preserved": len(startup), "certified_or_no_certified_runs_retried": 0}
    artifact["summary"] = aggregate(retained)
    write(AUDIT, artifact)


def freeze_slow():
    require_prepared()
    audit = read(AUDIT)
    if audit["status"] != "COMPLETE" or len(audit["records"]) != 72:
        raise ValueError("Stage A must be complete")
    if SLOW.exists():
        raise ValueError("slow set already frozen")
    calls = [c for r in audit["records"] for c in r["reference_trace"]]
    unique = {}
    for c in sorted(calls, key=lambda c: (-c["scheduler_duration"], c["call_identity"])):
        unique.setdefault(c["call_identity"], c)
    ordered = list(unique.values())
    selected = {c["call_identity"]: c for c in ordered[:20]}
    categories = {"FEASIBLE": lambda c:c["schedule_status"] == "FEASIBLE",
        "RECOVERED_FEASIBLE": lambda c:c["recovered"], "REMAINING_DEADLOCK": lambda c:c["remaining_deadlock"]}
    available = {}
    for name, predicate in categories.items():
        category = [c for c in ordered if predicate(c)]
        available[name] = len(category)
        for c in category[:5]:
            selected[c["call_identity"]] = c
    frozen = {"slow_call_set_id": "PHASE3ZR_SLOW_CALL_SET_V1", "created_at": now(),
        "calls": list(selected.values()), "available_status_counts": available}
    frozen["slow_call_set_hash"] = digest(frozen)
    write(SLOW, {"frozen_set": frozen, "profiles": [], "status": "FROZEN"})
    print(f"frozen {len(selected)} slow calls; categories {available}", flush=True)


def replay_call(call, level=2, with_cprofile=False):
    solution, config, directions = reconstruct(call["replay"])
    if solution.canonical_hash != call["solution_canonical_hash"]:
        raise ValueError("candidate reconstruction differs")
    entry = {k: call[k] for k in ("instance_id", "instance_geometry_hash", "tier")}
    profiler = cProfile.Profile() if with_cprofile else None
    with ReferenceTrace(call["method_id"], level) as trace:
        if profiler is not None:
            profiler.enable()
        try:
            result = scheduler.FormalReferenceEvaluator(FORMAL_SCOPE_V2)(solution, config, orientations=directions)
        finally:
            if profiler is not None:
                profiler.disable()
        certification = certify_schedule(solution, result, config, scope=FORMAL_SCOPE_V2) if result.feasible else None
    row = trace.serialize(entry)[0]
    row["certified"] = bool(certification and certification.certified)
    row["observational_equivalence"] = all(row[k] == call[k] for k in ("schedule_status", "Cmax", "baseline_deadlock", "schedule_source", "recovery_rollouts", "rollout_budget", "max_discrepancies_used", "branch_points_considered", "frontier_exhausted"))
    row["canonical_schedule"] = result.canonical_json()
    if profiler is not None:
        stats = pstats.Stats(profiler)
        row["cprofile_top_cumulative"] = [{"function": f"{key[0]}:{key[1]}:{key[2]}",
            "primitive_calls": v[0], "calls": v[1], "self_time": v[2], "cumulative_time": v[3]}
            for key,v in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:30]]
        row["cprofile_top_self"] = [{"function": f"{key[0]}:{key[1]}:{key[2]}",
            "calls": v[1], "self_time": v[2], "cumulative_time": v[3]}
            for key,v in sorted(stats.stats.items(), key=lambda item: -item[1][2])[:20]]
    return row


def stage_b():
    require_prepared()
    artifact = read(SLOW)
    frozen = artifact["frozen_set"]
    if digest({k:v for k,v in frozen.items() if k != "slow_call_set_hash"}) != frozen["slow_call_set_hash"]:
        raise ValueError("frozen slow-call set changed")
    done = {p["call_identity"] for p in artifact["profiles"]}
    for call in artifact["frozen_set"]["calls"]:
        if call["call_identity"] in done:
            continue
        print(f"PROFILE {len(artifact['profiles'])+1}/{len(artifact['frozen_set']['calls'])} original={call['scheduler_duration']:.3f}", flush=True)
        row = replay_call(call, with_cprofile=len(artifact["profiles"]) < 3)
        artifact["profiles"].append(row)
        write(SLOW, artifact)
        print(f"PROFILE_DONE duration={row['scheduler_duration']:.3f} baseline={row['baseline_dispatch_time']:.3f} recovery={row['recovery_time']:.3f}", flush=True)
    profiles = artifact["profiles"]
    baseline = sum(p["baseline_dispatch_time"] for p in profiles)
    recovery = sum(p["recovery_time"] for p in profiles)
    artifact["diagnosis"] = {"dominant_component": "B32_RECOVERY" if recovery > baseline else "BASELINE_DISPATCH",
        "baseline_total": baseline, "recovery_total": recovery,
        "per_rollout_distribution": distribution([r["duration"] for p in profiles for r in p["rollouts"]]),
        "observational_equivalence": all(p["observational_equivalence"] for p in profiles),
        "performance_threshold_eligible": False}
    artifact["status"] = "COMPLETE"
    write(SLOW, artifact)


def hga_audit():
    p, s = require_prepared()
    if read(SLOW)["status"] != "COMPLETE":
        raise ValueError("Stage B diagnosis must precede HGA audit")
    artifact = read(HGA) if HGA.exists() else {"protocol_hash": p["protocol_hash"], "records": [], "status": "RUNNING"}
    if artifact["protocol_hash"] != p["protocol_hash"]:
        raise ValueError("HGA resume protocol differs")
    completed = {(r["instance_id"], r["solver_seed"]) for r in artifact["records"]}
    if len(completed) != len(artifact["records"]):
        raise ValueError("duplicate HGA run records")
    stage = {(r["instance_id"], r["solver_seed"]): r for r in read(AUDIT)["records"] if r["method_id"] == METHODS[1]}
    source = provenance()
    for row in stage.values():
        if (row["provenance"]["source_tree_hash"] != source["source_tree_hash"]
            or row["scope_hash"] != p["scope_hash"] or row["runtime_set_hash"] != s["runtime_set_hash"]
            or row["method_config"] != p["methods"][METHODS[1]]["config"]):
            raise ValueError("HGA reuse requires exactly matching scientific identity")
    for entry in (e for e in s["instances"] if e["tier"] == "LARGE"):
        for seed in HGA_SEEDS:
            key = (entry["instance_id"], seed)
            if key in completed:
                continue
            print(f"HGA {len(artifact['records'])+1}/40 {entry['N']} {seed}", flush=True)
            row = stage[key] if key in stage else run_one(entry, METHODS[1], seed)
            artifact["records"].append({**row, "reused_stage_a": key in stage})
            artifact["summary"] = aggregate(artifact["records"])
            write(HGA, artifact)
            completed.add(key)
    count = sum(r["final_certified"] for r in artifact["records"])
    artifact["classification"] = "NOT_EVALUABLE" if len(artifact["records"]) != 40 or artifact["summary"]["execution_failure"] else "ROBUST_95" if count >=38 else "INTERMITTENT_NO_CERTIFIED" if count>=32 else "SYSTEMIC_NO_CERTIFIED"
    artifact["status"] = "COMPLETE"
    write(HGA, artifact)


def forensic():
    if read(SLOW)["status"] != "COMPLETE" or read(HGA)["status"] != "COMPLETE":
        raise ValueError("development diagnosis and HGA audit required before external forensic")
    artifact = read(AUDIT)
    if "external_frozen_stress" in artifact:
        raise ValueError("forensic already recorded; no retry")
    entry = read(historical.PROTOCOL_PATH)["instances"][10]
    artifact["external_frozen_stress"] = run_one(entry, METHODS[1], 20261015, forensic=True)
    write(AUDIT, artifact)


def runtime_gate(summary):
    d = summary["reference_distribution"]["ALL"]
    return (summary["runs"] == 72 and d["count"] > 0 and all(d[k] <= THRESHOLDS[k] for k in ("p95", "p99", "max"))
        and summary["run_runtime"]["max"] <= THRESHOLDS["run_max"] and summary["numeric_failure"] == 0
        and summary["certifier_mismatch"] == 0 and summary["execution_failure"] == 0)


def summarize():
    audit, slow, hga = read(AUDIT), read(SLOW), read(HGA)
    passed = runtime_gate(audit["summary"])
    print(json.dumps({"runtime_gate": passed, "summary": audit["summary"], "diagnosis": slow.get("diagnosis"),
        "hga_class": hga.get("classification")}, indent=2), flush=True)


def assess_decision(audit, slow, hga, runtime_set, *, regression_pass, historical_unchanged):
    expected_a = {(e["instance_id"], m, seed) for e in runtime_set["instances"] for m in METHODS for seed in SEEDS}
    expected_h = {(e["instance_id"], seed) for e in runtime_set["instances"] if e["tier"] == "LARGE" for seed in HGA_SEEDS}
    rows_a, rows_h = audit["records"], hga["records"]
    actual_a = {(r["instance_id"], r["method_id"], r["solver_seed"]) for r in rows_a}
    actual_h = {(r["instance_id"], r["solver_seed"]) for r in rows_h}
    h_summary = hga["summary"]
    h_runtime = h_summary["reference_distribution"]["ALL"]
    checks = {
        "stage_a_complete": audit["status"] == "COMPLETE" and len(rows_a) == 72 and actual_a == expected_a,
        "runtime_gate": runtime_gate(audit["summary"]),
        "deep_profiling_complete": slow["status"] == "COMPLETE" and len(slow["profiles"]) == len(slow["frozen_set"]["calls"]),
        "observer_semantics": bool(slow["profiles"]) and all(p["observational_equivalence"] for p in slow["profiles"]),
        "hga_complete": hga["status"] == "COMPLETE" and len(rows_h) == 40 and actual_h == expected_h and all(r["method_id"] == METHODS[1] for r in rows_h),
        "hga_execution_integrity": h_summary["numeric_failure"] == 0 and h_summary["certifier_mismatch"] == 0 and h_summary["execution_failure"] == 0,
        "hga_runtime_integrity": h_summary["run_runtime"]["max"] <= 120.0 and h_runtime["max"] is not None and h_runtime["max"] <= 60.0,
        "historical_evidence_unchanged": historical_unchanged,
        "full_regression": regression_pass,
        "external_forensic_complete": "external_frozen_stress" in audit and audit["external_frozen_stress"]["execution_status"] == "COMPLETED",
        "development_isolation": all(r["data_role"] == ROLE for r in rows_a+rows_h),
    }
    passed = all(checks.values())
    runtime_closed = all(checks[k] for k in ("stage_a_complete", "runtime_gate", "deep_profiling_complete", "observer_semantics", "hga_runtime_integrity"))
    return {"checks": checks,
        "PHASE3ZR_EXECUTION_STATUS": "PASS" if passed else "FAIL",
        "REFERENCE_RUNTIME_STATUS": "CLOSED_WITHOUT_CODE_CHANGE" if runtime_closed else "OPEN",
        "REFERENCE_SEMANTIC_EQUIVALENCE": "NOT_APPLICABLE",
        "HGA_CERTIFIED_INCUMBENT_STATUS": hga["classification"],
        "PHASE3Z_HISTORICAL_STATUS": "FAIL_179_OF_180_CERTIFIED",
        "DETERMINISTIC_V2_BACKBONE_STATUS": "NOT_EVALUABLE_IN_PHASE3Z",
        "TRADITIONAL_HEURISTIC_TUNING_STATUS": "CLOSED" if passed else "NOT_CLOSED",
        "V2_VALIDATION_STATUS": "CONSUMED", "ID_TEST_STATUS": "SEALED" if checks["development_isolation"] else "VIOLATED",
        "PHASE4_0_AUTHORIZED": "YES" if passed and runtime_closed else "NO",
        "NEXT_PHASE": "Phase 4-0 — V2 Candidate-Pool Oracle Recall Audit" if passed and runtime_closed else "Resolve Phase3-ZR remaining execution/runtime blocker only"}


def finalize():
    require_prepared()
    audit = read(AUDIT)
    # A directory migration must not rerun or rewrite the completed experiment.
    # The original historical-byte comparison is retained in its decision.
    if "finalized_at" in audit:
        print(audit["final_regression"]["stdout"], flush=True)
        print(json.dumps(audit["decision"], indent=2, ensure_ascii=False), flush=True)
        return
    slow, hga = read(SLOW), read(HGA)
    if not (audit["status"] == slow["status"] == hga["status"] == "COMPLETE") or "external_frozen_stress" not in audit:
        raise ValueError("all measurements must precede the final authorization decision")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    audit["final_regression"] = {"returncode": result.returncode, "stdout": result.stdout,
                               "stderr": result.stderr, "duration": time.perf_counter()-started, "at": now()}
    # Compare with the existing untouched handoff copy, using its bytes rather
    # than creating another history hash/baseline/contract.
    original_copy = ROOT.parents[1] / "data/validation/phase3z_v2_common_model_validation_v1.json"
    unchanged = original_copy.exists() and original_copy.read_bytes() == historical.ARTIFACT_PATH.read_bytes()
    audit["decision"] = assess_decision(audit, slow, hga, read(RUNTIME_SET),
        regression_pass=result.returncode == 0, historical_unchanged=unchanged)
    audit["finalized_at"] = now()
    write(AUDIT, audit)
    print(result.stdout, flush=True)
    print(json.dumps(audit["decision"], indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "stage-a", "repair-startup", "freeze-slow", "stage-b", "hga", "forensic", "summarize", "finalize", "compact-storage"))
    action = parser.parse_args().action
    {"prepare":prepare, "stage-a":stage_a, "repair-startup":repair_startup, "freeze-slow":freeze_slow, "stage-b":stage_b,
     "hga":hga_audit, "forensic":forensic, "summarize":summarize, "finalize":finalize, "compact-storage":compact_storage}[action]()


if __name__ == "__main__":
    main()
