"""Bounded deterministic finite-X audit; never a global X-domain oracle.

All production search, scientific geometry, direction, scheduling and certification
code is reused without modification. Stage A evaluates every insertion pair cheaply,
then at most 16 direction-DP and 4 formal candidates per frozen pattern.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

if __package__:
    from scripts import run_phase3x_xsplit_gate as historical
else:
    import run_phase3x_xsplit_gate as historical
from mrta_reference.candidate import apply_candidate
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import (
    blocks_for_pattern, finite_x_split_validator, frozen_handover_centers,
    generate_x_split_patterns, oriented_endpoints, robot_is_eligible,
    x_split_geometry_metadata,
)
from mrta_reference.model import (
    CandidateKey, CandidateMove, CanonicalSolution, MoveType, OperationKind,
    ParentWeld, Rail, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern,
)
from mrta_reference.provenance import compute_source_tree_hash
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import ACTIVE_FORMAL_SCOPE, EXPERIMENTAL_X_SPLIT_SCOPE_V1, FORMAL_SCOPE_V1_1
from mrta_reference.solution import block_map, canonicalize, official_metrics
from mrta_search.direction import (
    DirectionStatus, _initial_error, optimize_directions_with_initial_feasibility,
)
from mrta_search.neighborhood import _directed_proxy, _direction_hints
from mrta_search.pipeline import run_bounded_sa_oi

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/manifests/PHASE3X2_X_ORACLE_PROTOCOL_V1.json"
ARTIFACT_PATH = ROOT / "data/development/phase3x2_x_oracle_audit_v1.json"
GATE_HASH = "4bfe998a9caef567a66db24a77e9a598b24bda4be7967e51d0484c1548558c93"
SCOPE_HASH = "28236ee1caa75ce669af2a62ddbd28e2c557ed4b00eb5d4a1d8a29335aa209c8"
K_INSERT_DP = 16
K_INSERT_REF = 4
SOURCE_LABEL = "KNOWN_MAIN_f75ddc4044d9d066a6a5a89d91c03ef50c61386d+LOCAL_PHASE3X2_AUDIT"


class AuditFailure(RuntimeError):
    """Execution invalid: no scientific decision may be emitted."""


def digest(value: Any) -> str:
    return hashlib.sha256(historical._canonical_json(value).encode("utf-8")).hexdigest()


def scientific_payload(value: Any) -> Any:
    """Explicitly exclude wall clock and cache/profile counters from replay identity."""
    if isinstance(value, dict):
        return {
            key: scientific_payload(item) for key, item in value.items()
            if not key.endswith("_seconds") and key not in {"cache", "scientific_hash"}
        }
    if isinstance(value, (tuple, list)):
        return [scientific_payload(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    # Artifact generation writes only the individually approved target file.
    # Completed rows are saved immediately; a malformed interrupted write fails closed.
    if not path.parent.is_dir():
        raise AuditFailure(f"output directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def assert_development_roles(split: dict, paths) -> None:
    roles = {row["relative_path"]: row["assigned_role"] for row in split["workbooks"]}
    historical.assert_development_access(split, tuple(paths))
    if any(roles.get(path) != "DEVELOPMENT_CONSUMED" for path in paths):
        raise PermissionError("Phase 3-X2 permits DEVELOPMENT_CONSUMED only")


def pattern_census(entry, parents, config) -> list[dict]:
    centers = frozen_handover_centers(parents, config)
    rows = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        for rail, center in ((Rail.UPPER, centers[0]), (Rail.LOWER, centers[1])):
            for pattern in generate_x_split_patterns(parent, center, config, rail=rail):
                children = blocks_for_pattern(parent, pattern, config)
                span = abs(parent.end[0] - parent.start[0])
                processing = config.process_time(parent.length)
                rows.append({
                    "instance_id": entry["instance_id"], "pattern_id": pattern.pattern_id,
                    "parent_id": parent.parent_id, "rail": rail.value, "t": pattern.t,
                    "point_id": pattern.point_id, "source": pattern.point_id,
                    "parent_length": parent.length, "child_lengths": [b.length for b in children],
                    "child_process_times": [config.process_time(b.length) for b in children],
                    "x_span": span, "whole_process_time": processing,
                    "extra_fixed_processing": config.t_pre + config.t_post,
                    "blocking_proxy": processing * span,
                    "blocking_proxy_norm": processing * span / (config.workspace_x[1] - config.workspace_x[0]),
                    "cross_x_g": (parent.start[0] - center) * (parent.end[0] - center)
                    < -(config.numeric_epsilon ** 2),
                    "midpoint_distance_to_x_g": abs(parent.point(0.5)[0] - center),
                    "x_g": center,
                })
    return sorted(rows, key=lambda row: (row["instance_id"], row["parent_id"], row["rail"], row["t"], row["pattern_id"]))


def pattern_from_row(row: dict) -> SplitPattern:
    return SplitPattern(row["parent_id"], SplitKind.X_SPLIT, row["t"], row["point_id"], rail=Rail(row["rail"]))


def solution_from_payload(payload: dict, config: ScientificConfig) -> CanonicalSolution:
    parents = tuple(ParentWeld(row[0], tuple(row[1]), tuple(row[2])) for row in payload["parents"])
    patterns = tuple(
        SplitPattern(row[0], SplitKind(row[1]), row[2], row[3], row[4],
                     Rail(row[5]) if len(row) > 5 else None)
        for row in payload["patterns"]
    )
    return canonicalize(parents, patterns, {row[0]: tuple(row[1]) for row in payload["routes"]},
                        config, x_split_validator=finite_x_split_validator(parents, config))


def solution_metrics(solution, schedule, config) -> dict:
    metrics = official_metrics(solution, schedule, config)
    blocks = block_map(solution, config)
    loads = [sum(config.process_time(blocks[b].length) for b in route.block_ids) for route in solution.routes]
    waits = [sum(op.duration for op in schedule.operations if op.robot_id == r and op.kind is OperationKind.WAIT) for r in range(4)]
    empty = []
    for r in range(4):
        welds = sorted((op for op in schedule.operations if op.robot_id == r and op.kind is OperationKind.WELD), key=lambda op: op.sequence_index)
        empty.append(sum(math.dist(a.end, b.start) / config.empty_speed for a, b in zip(welds, welds[1:])))
    makespan = max(range(4), key=lambda r: (schedule.robot_completion[r], -r))
    return {
        "cmax": metrics.cmax, "robot_process_time": loads, "process_spread": metrics.process_imbalance,
        "robot_empty_travel": empty, "empty_travel": metrics.total_empty_travel,
        "robot_wait": waits, "total_wait": metrics.total_waiting,
        "makespan_robot_id": makespan, "makespan_robot_wait": waits[makespan],
        "robot_completion": list(schedule.robot_completion), "reference_status": schedule.status.value,
        "baseline_deadlock": schedule.baseline_deadlock, "reference_source": schedule.source,
        "recovery_rollouts": schedule.recovery_rollouts, "rollout_budget": schedule.rollout_budget,
        "scope_hash": schedule.scope_hash, "reference_policy_id": schedule.reference_policy_id,
        "x_split_parent_count": sum(p.kind is SplitKind.X_SPLIT for p in solution.patterns),
    }


def build_context(ppo_root: Path) -> dict:
    config = ScientificConfig()
    gate = historical._read_json(historical.GATE_SET_PATH)
    history = historical._read_json(historical.ARTIFACT_PATH)
    split = historical._read_json(historical.SPLIT_PATH)
    dataset = historical._read_json(historical.DATASET_PATH)
    old_protocol = historical._read_json(historical.PROTOCOL_PATH)
    historical.validate_gate_set(gate, dataset, split)
    if gate["xsplit_gate_set_hash"] != GATE_HASH or EXPERIMENTAL_X_SPLIT_SCOPE_V1.scope_hash != SCOPE_HASH:
        raise AuditFailure("frozen gate set or experimental scope mismatch")
    if old_protocol != historical.build_protocol(gate) or history["protocol_hash"] != old_protocol["phase3x_protocol_hash"]:
        raise AuditFailure("historical Phase 3-X protocol mismatch")
    if len(history["records"]) != 72 or history["gate_decision"]["X_SPLIT_MECHANISM_STATUS"] != "NOT_SUPPORTED":
        raise AuditFailure("historical Phase 3-X evidence mismatch")
    if ACTIVE_FORMAL_SCOPE != FORMAL_SCOPE_V1_1:
        raise AuditFailure("active formal scope is not V1.1")
    entries = gate["instances"]
    assert_development_roles(split, [e["relative_path"] for e in entries])
    census = []
    instances = {}
    for entry in entries:
        instance = historical._load_instance(entry, split, ppo_root)
        parents = historical.to_parent_welds(instance)
        if x_split_geometry_metadata(parents, config) != entry["x_geometry_metadata"]:
            raise AuditFailure(f"frozen X metadata mismatch: {entry['instance_id']}")
        rows = pattern_census(entry, parents, config)
        if len({(r["parent_id"], r["rail"], r["t"]) for r in rows}) != len(rows):
            raise AuditFailure("duplicate scientific X pattern")
        common, seconds = historical._common_seed(parents, source_commit_label=SOURCE_LABEL)
        previous = history["common_seeds"][entry["instance_id"]]
        if common.solution.canonical_hash != previous["common_seed_hash"] or common.schedule.cmax != previous["common_seed_cmax"]:
            raise AuditFailure(f"common seed hash/Cmax mismatch: {entry['instance_id']}")
        kinds = {p.parent_id: p.kind for p in common.solution.patterns}
        if any(kinds[r["parent_id"]] is not SplitKind.WHOLE for r in rows):
            raise AuditFailure("frozen X parent is not WHOLE in the unchanged common seed")
        census.extend(rows)
        instances[entry["instance_id"]] = {
            "entry": entry, "parents": parents, "common": common, "construction_seconds": seconds,
            "census": rows, "common_metrics": solution_metrics(common.solution, common.schedule, config),
        }
        print(f"verified common seed and {len(rows)} X patterns: {entry['instance_id']}", flush=True)
    if len(census) != 539:
        raise AuditFailure(f"expected 539 X patterns, got {len(census)}")
    # Old manifest records aggregates, not individual pattern IDs. Their equivalence
    # is established by unchanged scientific source, geometry and complete metadata.
    if compute_source_tree_hash(ROOT) != history["provenance"]["source_tree_hash"]:
        raise AuditFailure("scientific source changed since historical Phase 3-X")
    census_payload = {"census_id": "X_PATTERN_CENSUS_V1", "patterns": census}
    return {"gate": gate, "history": history, "instances": instances,
            "census": census_payload, "census_hash": digest(census_payload)}


def build_protocol(context: dict) -> dict:
    payload = {
        "protocol_id": "PHASE3X2_X_ORACLE_PROTOCOL_V1",
        "audit_name": "BOUNDED_DETERMINISTIC_X_PATTERN_POTENTIAL_AUDIT",
        "not_global_or_exact_X_optimization": True,
        "gate_set_hash": GATE_HASH, "experimental_scope_hash": SCOPE_HASH,
        "scientific_config_hash": ScientificConfig().scientific_hash,
        "scientific_source_tree_hash": compute_source_tree_hash(ROOT),
        "runner_sha256": historical._sha256_file(Path(__file__)),
        "source_commit_label": SOURCE_LABEL, "source_commit_verified": False,
        "historical_artifact_sha256": historical._sha256_file(historical.ARTIFACT_PATH),
        "historical_protocol_hash": context["history"]["protocol_hash"],
        "x_pattern_census": context["census"], "x_pattern_census_hash": context["census_hash"],
        "common_seeds": {
            key: {"hash": info["common"].solution.canonical_hash, "cmax": info["common"].schedule.cmax}
            for key, info in context["instances"].items()
        },
        "K_INSERT_DP": K_INSERT_DP, "K_INSERT_REF": K_INSERT_REF,
        "reference_upper_bound": 2156,
        "cheap_rank": ["max_robot_process_load", "process_spread", "existing_common_direction_empty_proxy", "existing_zero_direction_empty_proxy", "canonical_solution_hash"],
        "new_child_proxy_direction": 0,
        "direction_rank": ["direction_DP_total_empty_travel", "cheap_rank", "canonical_solution_hash"],
        "all_insertion_pairs_cheap_enumerated": True,
        "first_orientation_existence_checked_if_top16_all_infeasible": True,
        "feasible_outside_top16_conflict_policy": "EXECUTION_FAIL_WITHOUT_CHANGING_CAPS",
        "StageA_support_rule": {"different_workbooks_ge_1_percent": 2, "one_supporting_N_ge": 50, "one_supporting_best_x_span_ge": 2.0, "all_supporting_certified": True},
        "StageB_trigger": "if_and_only_if_X_ONE_STEP_POTENTIAL_PRESENT",
        "StageB_seeds": list(historical.SEEDS), "StageB_search_seconds": 60.0,
        "StageB_search_config": json.loads(historical._canonical_json(asdict(historical.gate_search_config()))),
        "StageB_rule": {"different_workbooks": 2, "winning_seeds_out_of_3": 2, "median_improvement_ge": 0.01, "supporting_final_X_exists": True},
        "final_rule": {"NOT_ESTABLISHED": "CASE1_NO_V2", "PRESENT_WEAK": "CASE2_V2_AUTHORIZED", "PRESENT_ADEQUATE": "CASE3_V2_AUTHORIZED", "EXECUTION_FAILURE": "CASE4_UNRESOLVED"},
        "forbidden_roles": ["TRAIN_POOL", "VALIDATION", "ID_TEST"],
        "deterministic_artifact_hash_excludes": ["fields_ending_in_seconds", "cache", "scientific_hash"],
        "historical_pattern_access": "UNKNOWN_UNLESS_PROVEN_BY_FINAL_PATTERN_RETENTION",
    }
    payload["protocol_hash"] = digest(payload)
    return payload


@dataclass(frozen=True)
class InsertionCandidate:
    solution: CanonicalSolution
    positions: tuple[int, int]
    cheap_rank: tuple


def insertion_grid(common: CanonicalSolution, pattern: SplitPattern) -> tuple[tuple[int, int], ...]:
    if pattern.kind is not SplitKind.X_SPLIT or pattern.rail is None:
        raise ValueError("audit requires a rail-specific X_SPLIT")
    existing = next(p for p in common.patterns if p.parent_id == pattern.parent_id)
    if existing.kind is not SplitKind.WHOLE:
        raise ValueError("one-step audit activates a WHOLE parent only")
    left, right = (0, 1) if pattern.rail is Rail.UPPER else (2, 3)
    whole = f"{pattern.parent_id}::whole"
    lengths = [sum(b != whole for b in common.routes[r].block_ids) for r in (left, right)]
    return tuple(itertools.product(range(lengths[0] + 1), range(lengths[1] + 1)))


def construct_insertion(common, pattern, positions, config, validator) -> CanonicalSolution:
    whole = f"{pattern.parent_id}::whole"
    source, source_position = next(
        (r.robot_id, i) for r in common.routes for i, b in enumerate(r.block_ids) if b == whole
    )
    left = 0 if pattern.rail is Rail.UPPER else 2
    move = CandidateMove(
        CandidateKey(common.revision, MoveType.SPLIT_ACTIVATE, (pattern.parent_id,), source,
                     left, (source_position,), positions, pattern.pattern_id, pattern.point_id),
        (whole,), pattern,
    )
    return apply_candidate(common, move, config, x_split_validator=validator)


def cheap_rank(solution, config, hints) -> tuple:
    blocks = block_map(solution, config)
    if any(not robot_is_eligible(blocks[b], route.robot_id, config)
           for route in solution.routes for b in route.block_ids):
        raise ValueError("robot eligibility failure")
    loads = [sum(config.process_time(blocks[b].length) for b in route.block_ids) for route in solution.routes]
    return (max(loads), max(loads) - min(loads), _directed_proxy(solution, hints, config),
            _directed_proxy(solution, {}, config), solution.canonical_hash)


def shortlist_cheap(candidates):
    return sorted(candidates, key=lambda c: c.cheap_rank)[:K_INSERT_DP]


def shortlist_direction(candidates):
    return sorted(candidates, key=lambda pair: (pair[1].total_empty_travel, pair[0].cheap_rank, pair[0].solution.canonical_hash))[:K_INSERT_REF]


def first_orientation_exists(solution, config) -> bool:
    # Same first-orientation predicate used by the existing DP; no route DP call.
    blocks = block_map(solution, config)
    active = [r.robot_id for r in solution.routes if r.block_ids]
    for bits in itertools.product((0, 1), repeat=len(active)):
        points = {r: oriented_endpoints(blocks[solution.routes[r].block_ids[0]], bit)[0] for r, bit in zip(active, bits)}
        if _initial_error(points, config) is None:
            return True
    return False


class AuditCache:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.dp = {}
        self.reference = {}
        self.hits = Counter()

    def direction(self, candidate, config, evaluator):
        key = (config.scientific_hash, candidate.solution.canonical_hash)
        if self.enabled and key in self.dp:
            self.hits["direction"] += 1
            return self.dp[key]
        result = evaluator(candidate.solution, config)
        if self.enabled:
            self.dp[key] = result
        return result

    def formal(self, candidate, direction, config, evaluator, certifier):
        key = (SCOPE_HASH, config.scientific_hash, candidate.solution.canonical_hash, direction.directions)
        if self.enabled and key in self.reference:
            self.hits["reference"] += 1
            return self.reference[key]
        schedule = evaluator(candidate.solution, config, scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
                             orientations={r: direction.directions[r] for r in range(4)})
        if schedule.status is ScheduleStatus.NUMERIC_FAILURE:
            raise AuditFailure("NUMERIC_FAILURE in deterministic X audit")
        expected = EXPERIMENTAL_X_SPLIT_SCOPE_V1
        if schedule.scope_hash != expected.scope_hash or schedule.reference_policy_id != expected.reference_scheduler_policy_id:
            raise AuditFailure("reference evaluator scope/policy mismatch")
        certificate = None
        if schedule.feasible:
            certificate = certifier(candidate.solution, schedule, config, scope=expected)
            if not certificate.certified:
                raise AuditFailure(f"scheduler FEASIBLE but independent certifier FAIL: {certificate.errors}")
        value = (schedule, certificate)
        if self.enabled:
            self.reference[key] = value
        return value


def audit_pattern(info, row, cache=None, *, direction_evaluator=optimize_directions_with_initial_feasibility,
                  reference_evaluator=reference_schedule_formal, certifier=certify_schedule) -> dict:
    started = time.perf_counter()
    config = ScientificConfig()
    cache = AuditCache() if cache is None else cache
    before_hits = dict(cache.hits)
    common = info["common"]
    solution = common.solution
    unchanged = solution.canonical_json
    pattern = pattern_from_row(row)
    validator = finite_x_split_validator(solution.parents, config)
    hints = _direction_hints(solution, common.directions)
    candidates = []
    canonical_valid = 0
    rejection_counts = Counter()
    grid = insertion_grid(solution, pattern)
    for positions in grid:
        try:
            provisional = construct_insertion(solution, pattern, positions, config, validator)
        except ValueError as error:
            rejection_counts[f"canonical:{error}"] += 1
            continue
        canonical_valid += 1
        try:
            rank = cheap_rank(provisional, config, hints)
        except ValueError as error:
            rejection_counts[f"cheap:{error}"] += 1
            continue
        candidates.append(InsertionCandidate(provisional, positions, rank))
    cheap_seconds = time.perf_counter() - started
    dp_candidates = shortlist_cheap(candidates)
    dp_started = time.perf_counter()
    directions = [(candidate, cache.direction(candidate, config, direction_evaluator)) for candidate in dp_candidates]
    feasible = [(candidate, direction) for candidate, direction in directions if direction.status is DirectionStatus.FEASIBLE]
    if not feasible and any(first_orientation_exists(candidate.solution, config) for candidate in candidates):
        raise AuditFailure("direction-feasible insertion outside frozen top16; cannot silently deny formal access or exceed cap")
    dp_seconds = time.perf_counter() - dp_started
    formal_candidates = shortlist_direction(feasible)
    if feasible and not formal_candidates:
        raise AuditFailure("direction-feasible pattern received no formal access")
    references = []
    best = None
    ref_started = time.perf_counter()
    for candidate, direction in formal_candidates:
        call_started = time.perf_counter()
        schedule, certificate = cache.formal(candidate, direction, config, reference_evaluator, certifier)
        record = {"insertion_positions": list(candidate.positions), "directions": [list(v) for v in direction.directions],
                  "solution_hash": candidate.solution.canonical_hash, "reference_status": schedule.status.value,
                  "cmax": schedule.cmax, "certified": bool(certificate and certificate.certified),
                  "baseline_deadlock": schedule.baseline_deadlock, "recovery_rollouts": schedule.recovery_rollouts,
                  "reference_source": schedule.source, "diagnostics": list(schedule.diagnostics),
                  "reference_seconds": time.perf_counter() - call_started}
        references.append(record)
        if certificate and certificate.certified:
            metrics = solution_metrics(candidate.solution, schedule, config)
            evidence = {**record, "metrics": metrics, "canonical_solution": candidate.solution.canonical_payload()}
            rank = (schedule.cmax, candidate.solution.canonical_hash, direction.directions)
            if best is None or rank < best[0]:
                best = (rank, evidence)
    ref_seconds = time.perf_counter() - ref_started
    if solution.canonical_json != unchanged:
        raise AuditFailure("audit mutated the common seed")
    status_counts = Counter(r["reference_status"] for r in references)
    best_evidence = None if best is None else best[1]
    return {
        **row, "total_insertion_pairs": len(grid), "canonical_valid_pairs": canonical_valid,
        "cheap_valid_pairs": len(candidates), "cheap_scored_pairs": len(candidates),
        "rejection_counts": dict(rejection_counts), "dp_evaluated_pairs": len(dp_candidates),
        "direction_feasible_pairs": len(feasible), "reference_evaluated_pairs": len(references),
        "reference_status_counts": {status.value: status_counts[status.value] for status in ScheduleStatus},
        "dp_shortlist": [{"insertion_positions": list(c.positions), "cheap_rank": list(c.cheap_rank),
                          "status": d.status.value, "objective": d.total_empty_travel} for c, d in directions],
        "formal_candidates": references, "best": best_evidence,
        "best_certified_cmax": None if best is None else best[0][0],
        "delta_vs_common_seed": None if best is None else best[0][0] - common.schedule.cmax,
        "improvement_ratio_vs_common_seed": None if best is None else (common.schedule.cmax - best[0][0]) / common.schedule.cmax,
        "cheap_seconds": cheap_seconds, "direction_seconds": dp_seconds, "reference_seconds": ref_seconds,
        "audit_seconds": time.perf_counter() - started,
        "cache": {key: count - before_hits.get(key, 0) for key, count in cache.hits.items()},
    }


def stage_a_rule(instances: list[dict]) -> dict:
    supported = [r for r in instances if r.get("one_step_x_improvement") is not None and r["one_step_x_improvement"] >= 0.01 and r.get("best_x_certified")]
    checks = {
        "two_workbooks": len({r["workbook"] for r in supported}) >= 2,
        "one_N_ge_50": any(r["N"] >= 50 for r in supported),
        "one_best_x_span_ge_2": any(r["best_x_xspan"] >= 2.0 for r in supported),
    }
    return {"X_ONE_STEP_POTENTIAL": "PRESENT" if all(checks.values()) else "NOT_ESTABLISHED",
            "checks": checks, "supporting_instance_ids": [r["instance_id"] for r in supported]}


def summarize_stage_a(context, rows):
    instances = []
    for iid, info in context["instances"].items():
        entries = [r for r in rows if r["instance_id"] == iid]
        feasible = [r for r in entries if r["best"] is not None]
        best = min(feasible, key=lambda r: (r["best_certified_cmax"], r["best"]["solution_hash"], r["pattern_id"])) if feasible else None
        summary = {
            "instance_id": iid, "workbook": info["entry"]["relative_path"], "N": info["entry"]["N"],
            "tier": info["entry"]["tier"], "opportunity_group": info["entry"]["opportunity_group"],
            "pattern_count": len(entries), "expected_pattern_count": len(info["census"]),
            "common_seed_hash": info["common"].solution.canonical_hash,
            "common_seed_cmax": info["common"].schedule.cmax,
            "common_metrics": info["common_metrics"],
            "common_seed_construction_seconds": info["construction_seconds"],
            "audit_cost_seconds": sum(r["audit_seconds"] for r in entries),
            "C_X1": None if best is None else best["best_certified_cmax"],
            "one_step_x_improvement": None if best is None else best["improvement_ratio_vs_common_seed"],
            "best_x_certified": best is not None,
            "status": "NO_FEASIBLE_X_ONE_STEP" if best is None else "CERTIFIED_X_ONE_STEP",
        }
        if best:
            summary.update({"best_x_pattern": best["pattern_id"], "best_x_parent": best["parent_id"],
                            "best_x_rail": best["rail"], "best_x_t": best["t"], "best_x_source": best["source"],
                            "best_x_xspan": best["x_span"], "best_x_parent_length": best["parent_length"],
                            "best_x_blocking_proxy": best["blocking_proxy"],
                            "best_x_insertion": best["best"]["insertion_positions"],
                            "best_x_directions": best["best"]["directions"], "best": best["best"]})
            for metric in ("process_spread", "empty_travel", "total_wait", "makespan_robot_wait"):
                summary[f"delta_{metric}"] = best["best"]["metrics"][metric] - info["common_metrics"][metric]
        instances.append(summary)
    return instances


def stage_b_rule(records: list[dict]) -> dict:
    groups = {}
    for r in records:
        groups.setdefault(r["instance_id"], {})[(r["solver_seed"], r["arm"])] = r
    summaries = []
    for iid, group in sorted(groups.items()):
        pairs = []
        retained = False
        for seed in historical.SEEDS:
            control = group.get((seed, "NO_X_FROM_COMMON"))
            finite = group.get((seed, "FINITE_X_FROM_BEST_X"))
            if control is None or finite is None:
                continue
            a, b = control["cmax_at_60"], finite["cmax_at_60"]
            pairs.append({"solver_seed": seed, "cmax_no_x": a, "cmax_x_seeded": b,
                          "improvement_ratio": (a - b) / a,
                          "x_retained": finite["x_split_parent_count"] > 0})
            retained |= finite["x_split_parent_count"] > 0
        median = statistics.median(p["improvement_ratio"] for p in pairs) if len(pairs) == 3 else None
        summaries.append({"instance_id": iid, "workbook": next(iter(group.values()))["workbook"],
                          "pairs": pairs, "winning_seeds": sum(p["cmax_x_seeded"] < p["cmax_no_x"] for p in pairs),
                          "median_improvement": median, "supporting_final_X_exists": retained})
    supporting = [r for r in summaries if r["median_improvement"] is not None and r["median_improvement"] >= 0.01 and r["winning_seeds"] >= 2]
    adequate = len({r["workbook"] for r in supporting}) >= 2 and any(r["supporting_final_X_exists"] for r in supporting)
    return {"X_SEARCH_RETENTION": "ADEQUATE" if adequate else "WEAK", "instances": summaries}


def final_decision(stage_a: dict, stage_b: dict | None, *, failed=False, complete=True) -> dict:
    if failed or not complete:
        case, potential, retention, value, v2 = 4, "NOT_EVALUABLE", "NOT_APPLICABLE", "UNRESOLVED", "NO"
        next_phase = "Phase 3-X2 — resolve invalid/incomplete execution before scientific decision"
    elif stage_a["X_ONE_STEP_POTENTIAL"] == "NOT_ESTABLISHED":
        case, potential, retention, value, v2 = 1, "NOT_ESTABLISHED", "NOT_APPLICABLE", "NOT_SUPPORTED_BY_BOUNDED_ORACLE_AUDIT", "NO"
        next_phase = "Phase 4-0 — Candidate-Pool Oracle Recall Audit under FORMAL_SCOPE_V1_1"
    elif stage_b is None:
        case, potential, retention, value, v2 = 4, "PRESENT", "NOT_APPLICABLE", "UNRESOLVED", "NO"
        next_phase = "Phase 3-X2 — complete triggered Stage B retention audit"
    else:
        potential, retention, v2 = "PRESENT", stage_b["X_SEARCH_RETENTION"], "YES"
        case = 3 if retention == "ADEQUATE" else 2
        value = "SUPPORTED" if case == 3 else "SUPPORTED_BUT_ACCESS_LIMITED"
        next_phase = "Phase 3-Y — FORMAL_SCOPE_V2 Core Closure" + (" + common-domain HGA/WAG/ALNS adaptation" if case == 3 else " + future candidate ranking must include X")
    return {"case": case, "PHASE3X2_EXECUTION_STATUS": "FAIL" if case == 4 else "PASS",
            "X_ONE_STEP_POTENTIAL": potential, "X_SEARCH_RETENTION": retention,
            "X_DOMAIN_VALUE_STATUS": value, "FORMAL_SCOPE_V2_AUTHORIZED": v2,
            "ACTIVE_FORMAL_SCOPE": ACTIVE_FORMAL_SCOPE.scope_id, "ID_TEST_STATUS": "SEALED", "NEXT_PHASE": next_phase}


def parent_explanations(context, rows):
    output = []
    history = context["history"]["records"]
    for iid, info in context["instances"].items():
        for parent in info["parents"]:
            candidates = [r for r in rows if r["instance_id"] == iid and r["parent_id"] == parent.parent_id]
            feasible = [r for r in candidates if r["best"] is not None]
            best = min(feasible, key=lambda r: (r["best_certified_cmax"], r["pattern_id"])) if feasible else None
            retained = [r["solver_seed"] for r in history if r["instance_id"] == iid and any(p["parent_id"] == parent.parent_id for p in r["chosen_x_patterns"])]
            span = abs(parent.end[0] - parent.start[0])
            output.append({"instance_id": iid, "parent_id": parent.parent_id,
                           "length": parent.length, "x_span": span,
                           "whole_process_time": ScientificConfig().process_time(parent.length),
                           "blocking_proxy": ScientificConfig().process_time(parent.length) * span,
                           "has_x_pattern": bool(candidates), "pattern_count": len(candidates),
                           "best_x_status": "FEASIBLE_CERTIFIED" if best else "NO_CERTIFIED_X_IN_BOUNDED_AUDIT",
                           "best_x_pattern": None if best is None else best["pattern_id"],
                           "delta_vs_common_seed": None if best is None else best["delta_vs_common_seed"],
                           "improvement_ratio": None if best is None else best["improvement_ratio_vs_common_seed"],
                           "historical_parent_access": "YES_PROVEN_BY_FINAL_RETENTION" if retained else "UNKNOWN_NOT_RECORDED",
                           "historical_retained_seeds": retained})
    return sorted(output, key=lambda r: (-r["blocking_proxy"], r["instance_id"], r["parent_id"]))


def refresh(artifact, context):
    rows = artifact["pattern_records"]
    instances = summarize_stage_a(context, rows)
    artifact["instance_summaries"] = instances
    artifact["stage_a"] = stage_a_rule(instances)
    complete_a = len(rows) == 539
    positive = [r for r in instances if r["one_step_x_improvement"] is not None and r["one_step_x_improvement"] > 0]
    expected_b = 6 * len(positive) if complete_a and artifact["stage_a"]["X_ONE_STEP_POTENTIAL"] == "PRESENT" else 0
    artifact["stage_b_expected_run_count"] = expected_b
    artifact["stage_b"] = stage_b_rule(artifact["stage_b_records"]) if expected_b and len(artifact["stage_b_records"]) == expected_b else None
    artifact["decision"] = final_decision(artifact["stage_a"], artifact["stage_b"], failed=bool(artifact.get("failure")), complete=complete_a)
    artifact["totals"] = {
        "patterns_completed": len(rows), "patterns_with_direction_feasible": sum(r["direction_feasible_pairs"] > 0 for r in rows),
        "patterns_certified_feasible": sum(r["best"] is not None for r in rows),
        "patterns_improving_common": sum(r["improvement_ratio_vs_common_seed"] is not None and r["improvement_ratio_vs_common_seed"] > 0 for r in rows),
        "patterns_improving_ge_1_percent": sum(r["improvement_ratio_vs_common_seed"] is not None and r["improvement_ratio_vs_common_seed"] >= 0.01 for r in rows),
        "insertion_pairs": sum(r["total_insertion_pairs"] for r in rows),
        "cheap_valid_pairs": sum(r["cheap_valid_pairs"] for r in rows),
        "direction_dp_evaluations": sum(r["dp_evaluated_pairs"] for r in rows),
        "reference_evaluations": sum(r["reference_evaluated_pairs"] for r in rows),
        "numeric_failures": sum(r["reference_status_counts"]["NUMERIC_FAILURE"] for r in rows),
        "audit_cost_seconds": sum(r["audit_seconds"] for r in rows),
    }
    if artifact["totals"]["reference_evaluations"] > 2156:
        raise AuditFailure("reference cap exceeded")
    explanations = parent_explanations(context, rows)
    artifact["top20_parents_by_blocking_proxy"] = explanations[:20]
    artifact["long_x_span_parents"] = [r for r in explanations if r["x_span"] >= 10.0]
    artifact["stage_a_scientific_hash"] = digest(scientific_payload({"pattern_records": rows}))


def verified_best_seed(info, summary):
    config = ScientificConfig()
    best = summary["best"]
    solution = solution_from_payload(best["canonical_solution"], config)
    directions = tuple(tuple(v) for v in best["directions"])
    schedule = reference_schedule_formal(solution, config, scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1, orientations={r: directions[r] for r in range(4)})
    certificate = certify_schedule(solution, schedule, config, scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1)
    if not certificate.certified or solution.canonical_hash != best["solution_hash"] or schedule.cmax != summary["C_X1"]:
        raise AuditFailure("frozen best X seed replay mismatch")
    return replace(info["common"], solution=solution, directions=directions, schedule=schedule,
                   certification=certificate, attempts=(), winning_strategy="FROZEN_DETERMINISTIC_X_ONE_STEP")


def run_stage_b(artifact, context):
    if artifact["stage_a"]["X_ONE_STEP_POTENTIAL"] != "PRESENT":
        return
    existing = {(r["instance_id"], r["solver_seed"], r["arm"]) for r in artifact["stage_b_records"]}
    for summary in artifact["instance_summaries"]:
        iid = summary["instance_id"]
        if summary["one_step_x_improvement"] is None or summary["one_step_x_improvement"] <= 0:
            artifact["stage_b_not_applicable"][iid] = "NOT_APPLICABLE_NO_POSITIVE_X_SEED"
            continue
        info = context["instances"][iid]
        best_seed = verified_best_seed(info, summary)
        for seed in historical.SEEDS:
            for arm in ("NO_X_FROM_COMMON", "FINITE_X_FROM_BEST_X"):
                key = (iid, seed, arm)
                if key in existing:
                    continue
                finite = arm == "FINITE_X_FROM_BEST_X"
                initialization = best_seed if finite else info["common"]
                result = run_bounded_sa_oi(info["parents"], ScientificConfig(), historical.gate_search_config(), seed=seed,
                                          scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1 if finite else FORMAL_SCOPE_V1_1,
                                          source_commit=SOURCE_LABEL, allow_unverified_source=True,
                                          initialization_override=initialization, enable_x_split=finite)
                record = historical._record(result, entry=info["entry"], arm=arm, seed=seed,
                                            common_seed_hash=info["common"].solution.canonical_hash,
                                            common_seed_cmax=info["common"].schedule.cmax,
                                            provenance={"protocol_hash": artifact["protocol_hash"], "evidence_role": "DEVELOPMENT_DIAGNOSTIC_NOT_FAIR_BENCHMARK"})
                chosen_ids = set(record["x_split_final_pattern_ids"])
                record.update({"initial_seed_hash": initialization.solution.canonical_hash,
                               "initial_seed_cmax": initialization.schedule.cmax, "best_x_seed_cmax": summary["C_X1"],
                               "search_only_60s_seconds": result.runtime,
                               "audit_cost_inclusive_seconds": result.runtime + summary["audit_cost_seconds"],
                               "global_best_updates_after_start": sum(cmax < initialization.schedule.cmax for _, cmax in result.stats.best_events),
                               "x_retention": "RETAINED" if summary["best_x_pattern"] in chosen_ids else "SWITCHED" if chosen_ids else "REMOVED"})
                if record["numeric_failure_count"] or not record["final_certified"] or record["final_cmax"] > initialization.schedule.cmax + ScientificConfig().numeric_epsilon:
                    raise AuditFailure("Stage B numeric/certifier/global-best retention failure")
                artifact["stage_b_records"].append(record)
                existing.add(key)
                refresh(artifact, context)
                write_json(ARTIFACT_PATH, artifact)
                print(f"Stage B {len(existing)}/{artifact['stage_b_expected_run_count']}: {iid} {seed} {arm}", flush=True)


def new_artifact(protocol):
    return {"artifact_id": "PHASE3X2_X_ORACLE_AUDIT_V1", "protocol_hash": protocol["protocol_hash"],
            "x_pattern_census_hash": protocol["x_pattern_census_hash"], "pattern_records": [],
            "stage_b_records": [], "stage_b_not_applicable": {}, "failure": None}


def validate_resume(artifact, protocol):
    if artifact["protocol_hash"] != protocol["protocol_hash"] or artifact["x_pattern_census_hash"] != protocol["x_pattern_census_hash"]:
        raise AuditFailure("resume protocol/census mismatch")
    keys = [(r["instance_id"], r["pattern_id"]) for r in artifact["pattern_records"]]
    expected = {(r["instance_id"], r["pattern_id"]): r for r in protocol["x_pattern_census"]["patterns"]}
    if len(keys) != len(set(keys)) or any(key not in expected for key in keys):
        raise AuditFailure("resume duplicate or unknown pattern")
    for row in artifact["pattern_records"]:
        frozen = expected[(row["instance_id"], row["pattern_id"])]
        if any(row[k] != v for k, v in frozen.items()) or row["dp_evaluated_pairs"] > 16 or row["reference_evaluated_pairs"] > 4:
            raise AuditFailure("resume pattern identity or cap mismatch")
    if artifact.get("failure"):
        raise AuditFailure("previous execution failed; resolve explicitly before resume")


def run_audit(context, protocol):
    artifact = historical._read_json(ARTIFACT_PATH) if ARTIFACT_PATH.exists() else new_artifact(protocol)
    validate_resume(artifact, protocol)
    existing = {(r["instance_id"], r["pattern_id"]) for r in artifact["pattern_records"]}
    try:
        for iid, info in context["instances"].items():
            cache = AuditCache()
            for row in info["census"]:
                if (iid, row["pattern_id"]) in existing:
                    continue
                result = audit_pattern(info, row, cache)
                artifact["pattern_records"].append(result)
                refresh(artifact, context)
                write_json(ARTIFACT_PATH, artifact)
                n = len(artifact["pattern_records"])
                print(f"Stage A {n}/539: {row['pattern_id']} ref={result['reference_evaluated_pairs']} certified={result['reference_status_counts']['FEASIBLE']}", flush=True)
        refresh(artifact, context)
        write_json(ARTIFACT_PATH, artifact)
        print("Stage A rule: " + historical._canonical_json(artifact["stage_a"]), flush=True)
        run_stage_b(artifact, context)
        refresh(artifact, context)
        write_json(ARTIFACT_PATH, artifact)
    except Exception as error:
        artifact["failure"] = f"{type(error).__name__}: {error}"
        refresh(artifact, context)
        write_json(ARTIFACT_PATH, artifact)
        raise
    return artifact


def verify_determinism(context, protocol):
    artifact = historical._read_json(ARTIFACT_PATH)
    validate_resume(artifact, protocol)
    if len(artifact["pattern_records"]) != 539:
        raise AuditFailure("full Stage A must complete before deterministic replay")
    expected = {(r["instance_id"], r["pattern_id"]): r for r in artifact["pattern_records"]}
    rows = []
    for iid, info in context["instances"].items():
        cache = AuditCache(enabled=False)
        for row in info["census"]:
            result = audit_pattern(info, row, cache)
            if scientific_payload(result) != scientific_payload(expected[(iid, row["pattern_id"])]):
                raise AuditFailure(f"deterministic replay mismatch: {iid} {row['pattern_id']}")
            rows.append(result)
            if len(rows) % 25 == 0:
                print(f"deterministic replay {len(rows)}/539", flush=True)
    replay_hash = digest(scientific_payload({"pattern_records": rows}))
    if replay_hash != artifact["stage_a_scientific_hash"]:
        raise AuditFailure("scientific artifact hash mismatch")
    artifact["determinism_verification"] = {"patterns_replayed": 539, "cache_disabled": True,
                                            "reference_evaluations": sum(r["reference_evaluated_pairs"] for r in rows),
                                            "scientific_hash": replay_hash, "match": True}
    write_json(ARTIFACT_PATH, artifact)
    print("deterministic replay PASS: " + replay_hash, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-root", type=Path, default=Path("D:/pybullet_test/MRTA_GA/ppo"))
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()
    context = build_context(args.ppo_root)
    protocol = build_protocol(context)
    if args.freeze_protocol:
        if PROTOCOL_PATH.exists() and historical._read_json(PROTOCOL_PATH) != protocol:
            raise AuditFailure("refusing to replace an existing frozen Phase 3-X2 protocol")
        write_json(PROTOCOL_PATH, protocol)
    if args.run or args.verify_determinism:
        if not PROTOCOL_PATH.exists() or historical._read_json(PROTOCOL_PATH) != protocol:
            raise AuditFailure("protocol must be frozen before any audit result")
    print("census_hash=" + context["census_hash"], flush=True)
    if args.run:
        artifact = run_audit(context, protocol)
        print(historical._canonical_json(artifact["decision"]), flush=True)
    if args.verify_determinism:
        verify_determinism(context, protocol)


if __name__ == "__main__":
    main()
