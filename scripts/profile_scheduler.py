from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
import runpy

from mrta_exact import ExactSolveStatus, exact_schedule_from_templates, solve_exact_micro
from mrta_reference.model import (
    CanonicalSolution,
    Operation,
    OperationKind,
    ParentWeld,
    Route,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.scheduler import (
    FormalReferenceEvaluator,
    SchedulerProfile,
    _bounded_dispatch_recovery,
    build_operation_templates,
    build_robot_routes,
    reference_schedule_from_templates_optimized,
    reference_schedule_from_templates_slow,
    reference_schedule_optimized,
    reference_schedule_slow,
)
from mrta_reference.certifier import certify_schedule
from mrta_reference.provenance import resolve_source_provenance
from mrta_reference.scope import FORMAL_SCOPE_V1
from mrta_search import SearchConfig, micro_gap_decomposition, run_bounded_sa_oi
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.initialization import InitializationStrategy, _construct, _patterns

from profile_phase2b1 import synthetic_parents


DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "development"
    / "f4_deadlock_stress_corpus.json"
)


def _development_provenance():
    return resolve_source_provenance(
        Path(__file__).resolve().parents[1],
        source_commit="UNVERIFIED_LOCAL_TREE",
        allow_unverified_source=True,
    )


def formal_scope_gate(stage: str, seeds: tuple[int, ...]) -> None:
    """Development-only freeze evidence; JSON lines on stdout, no dataset files."""
    from mrta_reference.scope import RunScientificIdentity
    from mrta_reference.scheduler import (
        reference_schedule_from_templates_formal, _bounded_dispatch_recovery,
    )
    from mrta_reference.certifier import certify_template_schedule
    scope = FORMAL_SCOPE_V1
    provenance = _development_provenance()
    def emit(row):
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    emit({
        "stage": "identity",
        **asdict(RunScientificIdentity.from_scope(scope, ScientificConfig(), provenance)),
        "development_only": True,
        "provenance_status": "UNVERIFIED_SOURCE_PROVENANCE",
    })
    fast = ScientificConfig(weld_speed=1, empty_speed=1, t_pre=1, t_post=1)
    if stage in ("all", "calibration"):
        fixtures = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_exact.py"))
        for budget in (16, 32, 64, 128):
            for name, templates in fixtures["_manual_oracle_cases"]().items():
                baseline = reference_schedule_from_templates_optimized(templates, fast)
                started = time.perf_counter()
                result = _bounded_dispatch_recovery(templates, fast, baseline, state_budget=budget)
                elapsed = time.perf_counter() - started
                certificate = certify_template_schedule(templates, result, fast) if result.feasible else None
                if result.feasible and not certificate.certified:
                    raise RuntimeError(certificate.errors)
                emit({"stage": "calibration", "case": name, "budget": budget,
                      "baseline": baseline.status.value, "status": result.status.value,
                      "expanded_states": result.expanded_states, "Cmax": result.cmax,
                      "certified": bool(certificate and certificate.certified), "runtime": elapsed,
                      "frontier_exhausted": result.frontier_exhausted})
        for name, templates in fixtures["_manual_oracle_cases"]().items():
            start = time.perf_counter()
            result = reference_schedule_from_templates_formal(templates, fast)
            emit({"stage": "E1-E4", "case": name, "status": result.status.value,
                  "Cmax": result.cmax, "source": result.source, "expanded_states": result.expanded_states,
                  "budget": result.state_budget, "runtime": time.perf_counter() - start,
                  "certified": certify_template_schedule(templates, result, fast).certified if result.feasible else None})
    if stage in ("all", "quality"):
        for name in ("Q1_assignment_trap", "Q2_route_order_trap", "Q3_direction_trap",
                     "Q4_optional_y_split_trap", "Q5_interference_wait_trap", "Q6_lns_basin_trap"):
            parents, seed, iterations = quality_fixture(name)
            exact = solve_exact_micro(parents, fast)
            result = run_bounded_sa_oi(parents, fast, SearchConfig(max_iterations=iterations),
                                      seed=seed, scope=scope, source_provenance=provenance)
            replay = run_bounded_sa_oi(
                parents,
                fast,
                SearchConfig(max_iterations=iterations),
                seed=seed,
                scope=scope,
                source_provenance=provenance,
            )
            if not result.final_certification or not result.final_certification.certified:
                raise RuntimeError(f"uncertified quality case {name}")
            reproducible = (
                replay.final_certification is not None
                and replay.final_certification.certified
                and result.best_solution.canonical_json == replay.best_solution.canonical_json
                and result.best_directions == replay.best_directions
                and result.best_schedule.canonical_json() == replay.best_schedule.canonical_json()
            )
            if not reproducible:
                raise RuntimeError(f"non-reproducible quality case {name}")
            emit({"stage": "quality", "case": name, "seed": seed,
                  "identity": asdict(result.stats.scientific_identity),
                  "development_C_star": exact.best_cmax, "development_exact_status": exact.status.value,
                  "formal_initial_Cref": result.initialization.schedule.cmax,
                  "formal_Cref": result.best_schedule.cmax, "formal_exact_gap": None,
                  "status": result.status.value, "certified": result.final_certification.certified,
                  "fixed_seed_reproducible": reproducible,
                  "runtime": result.runtime, "iterations": result.stats.iterations,
                  "nref": result.stats.nref, "direction_calls": result.stats.direction_refinement_calls,
                  "best_sources": result.stats.improvements_by_family,
                  "recoveries": sum(r["source"] == "BOUNDED_DEADLOCK_RECOVERY" for r in result.stats.reference_records)})
    if stage in ("all", "smoke"):
        for family in ("load_skew", "spatial_cluster", "handover_heavy", "interference_stress"):
            for size in (20, 50, 100):
                initial_json = None
                for seed in seeds:
                    result = run_bounded_sa_oi(development_family(family, size), ScientificConfig(),
                                              SearchConfig(max_iterations=2), seed=seed,
                                              scope=scope, source_provenance=provenance)
                    if not result.final_certification or not result.final_certification.certified:
                        raise RuntimeError(f"uncertified family case {family}/{size}/{seed}")
                    replay = result.initialization.schedule.canonical_json()
                    if initial_json is not None and initial_json != replay:
                        raise RuntimeError("deterministic initializer replay changed")
                    initial_json = replay
                    s = result.stats
                    if s.n_numeric_failure:
                        raise RuntimeError(f"numeric failure in {family}/{size}/{seed}")
                    emit({"stage": "smoke", "family": family, "N": size, "seed": seed,
                          "status": result.status.value, "certified": result.final_certification.certified,
                          "initial": result.initialization.schedule.cmax, "best": result.best_schedule.cmax,
                          "strategy": result.initialization.winning_strategy,
                          "runtime": result.runtime, "iterations": s.iterations, "nref": s.nref,
                          "direction_calls": s.direction_refinement_calls, "lns_attempts": s.attempted_by_family["LNS_REPAIRED"],
                          "lns_c4": s.c4_by_family["LNS_REPAIRED"], "lns_accepted": s.accepted_by_family["LNS_REPAIRED"],
                          "baseline_deadlocks": sum(r["baseline_deadlock"] for r in s.reference_records),
                          "recoveries": sum(r["source"] == "BOUNDED_DEADLOCK_RECOVERY" for r in s.reference_records),
                          "remaining_deadlocks": sum(
                              r["baseline_deadlock"] and r["status"] == "DEADLOCK"
                              for r in s.reference_records
                          ),
                          "initial_success": result.initialization.status.value == "SUCCESS",
                          "c4_feasible_rate": s.n_feasible / s.nref if s.nref else None,
                          "lns_c4_rate": (
                              s.c4_by_family["LNS_REPAIRED"] / s.attempted_by_family["LNS_REPAIRED"]
                              if s.attempted_by_family["LNS_REPAIRED"] else None
                          ),
                          "deadlock": s.n_deadlock, "repair_time": s.repair_time,
                          "reference_time": s.reference_scheduler_time, "certifier_time": s.certifier_time})


def _deadlock_identity(solution, directions, config: ScientificConfig) -> str:
    payload = [solution.canonical_hash, directions, config.scientific_hash]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class DeadlockCorpusCollector:
    def __init__(self):
        self.entries: dict[str, dict[str, object]] = {}
        self.origin: dict[str, object] | None = None
        self.raw_observations = 0

    def observe(self, solution, config, directions, baseline) -> None:
        if self.origin is None:
            raise RuntimeError("deadlock observation has no development origin")
        self.raw_observations += 1
        identity = _deadlock_identity(solution, directions, config)
        entry = self.entries.get(identity)
        origin = dict(self.origin)
        if entry is not None:
            if origin not in entry["origins"]:
                entry["origins"].append(origin)
            return
        canonical = solution.canonical_payload()
        baseline_json = baseline.canonical_json()
        self.entries[identity] = {
            "identity": identity,
            "solution_canonical_hash": solution.canonical_hash,
            "canonical_solution": canonical,
            "patterns": canonical["patterns"],
            "routes": canonical["routes"],
            "directions": [list(values) for values in directions],
            "scientific_config": config.scientific_mapping(),
            "scientific_config_hash": config.scientific_hash,
            "baseline": {
                "status": baseline.status.value,
                "diagnostics": list(baseline.diagnostics),
                "wait_for_graph": [list(edge) for edge in baseline.wait_for_graph],
                "canonical_hash": hashlib.sha256(baseline_json.encode("utf-8")).hexdigest(),
            },
            "origins": [origin],
        }


def collect_deadlock_stress_corpus(
    output_path: Path,
    *,
    seeds: tuple[int, ...],
    target_unique: int = 30,
) -> dict[str, object]:
    provenance = _development_provenance()
    collector = DeadlockCorpusCollector()
    evaluator = FormalReferenceEvaluator(FORMAL_SCOPE_V1, collector.observe)
    runs = []

    def execute(family: str, size: int, seed: int) -> None:
        collector.origin = {"family": family, "N": size, "seed": seed}
        result = run_bounded_sa_oi(
            development_family(family, size),
            ScientificConfig(),
            SearchConfig(max_iterations=2),
            seed=seed,
            scope=FORMAL_SCOPE_V1,
            reference_evaluator=evaluator,
            source_provenance=provenance,
        )
        if not result.final_certification or not result.final_certification.certified:
            raise RuntimeError(f"uncertified corpus collection run: {family}/{size}/{seed}")
        runs.append({
            "family": family,
            "N": size,
            "seed": seed,
            "status": result.status.value,
            "iterations": result.stats.iterations,
            "nref": result.stats.nref,
            "baseline_deadlocks": sum(row["baseline_deadlock"] for row in result.stats.reference_records),
            "recovered_deadlocks": sum(
                row["source"] == "BOUNDED_DEADLOCK_RECOVERY"
                for row in result.stats.reference_records
            ),
            "runtime": result.runtime,
        })

    for family in ("load_skew", "spatial_cluster", "handover_heavy", "interference_stress"):
        for size in (50, 100):
            for seed in seeds:
                execute(family, size, seed)
    extra_seed = max(seeds) + 1
    while len(collector.entries) < target_unique and extra_seed <= max(seeds) + 16:
        execute("handover_heavy", 100, extra_seed)
        extra_seed += 1

    corpus = {
        "schema_id": "F4_DEADLOCK_STRESS_CORPUS_V1",
        "development_only": True,
        "provenance_status": "UNVERIFIED_SOURCE_PROVENANCE",
        "source_provenance": asdict(provenance),
        "scope_id_at_collection": FORMAL_SCOPE_V1.scope_id,
        "scope_hash_at_collection": FORMAL_SCOPE_V1.scope_hash,
        "collection_policy": {
            "families": ["load_skew", "spatial_cluster", "handover_heavy", "interference_stress"],
            "sizes": [50, 100],
            "base_seeds": list(seeds),
            "iterations": 2,
            "target_unique": target_unique,
            "future_validation_test_ood_used": False,
        },
        "raw_deadlock_observations": collector.raw_observations,
        "unique_entries": len(collector.entries),
        "runs": runs,
        "entries": [collector.entries[key] for key in sorted(collector.entries)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return corpus


def _solution_from_payload(payload: dict[str, object]) -> CanonicalSolution:
    parents = tuple(ParentWeld(row[0], tuple(row[1]), tuple(row[2])) for row in payload["parents"])
    patterns = tuple(
        SplitPattern(row[0], SplitKind(row[1]), row[2], row[3], row[4])
        for row in payload["patterns"]
    )
    routes = tuple(Route(row[0], tuple(row[1])) for row in payload["routes"])
    solution = CanonicalSolution(parents, patterns, routes)
    if solution.canonical_payload() != payload:
        raise RuntimeError("corpus solution did not round-trip canonically")
    return solution


def _config_from_mapping(payload: dict[str, object]) -> ScientificConfig:
    values = dict(payload)
    values["workspace_x"] = tuple(values["workspace_x"])
    values["workspace_y"] = tuple(values["workspace_y"])
    return ScientificConfig(**values)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def detect_local_plateaus(
    entries: list[dict[str, object]],
    budgets: tuple[int, ...],
    *,
    tolerance: float = 1.0e-9,
) -> list[int]:
    candidates = []
    available = set(budgets)
    for budget in budgets:
        if 2 * budget not in available or 4 * budget not in available:
            continue
        stable = True
        for entry in entries:
            rows = [entry["replays"][str(value)] for value in (budget, 2 * budget, 4 * budget)]
            if len({row["status"] for row in rows}) != 1:
                stable = False
                break
            if rows[0]["status"] == ScheduleStatus.FEASIBLE.value:
                if not all(row["certified"] for row in rows):
                    stable = False
                    break
                if max(row["Cmax"] for row in rows) - min(row["Cmax"] for row in rows) > tolerance:
                    stable = False
                    break
        if stable:
            candidates.append(budget)
    return candidates


def replay_deadlock_stress_corpus(
    corpus_path: Path,
    *,
    budgets: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048),
) -> dict[str, object]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus["schema_id"] != "F4_DEADLOCK_STRESS_CORPUS_V1":
        raise RuntimeError("unsupported stress corpus schema")
    corpus["replay_source_provenance"] = asdict(_development_provenance())
    baseline_runtimes = []
    for entry in corpus["entries"]:
        solution = _solution_from_payload(entry["canonical_solution"])
        directions = tuple(tuple(row) for row in entry["directions"])
        config = _config_from_mapping(entry["scientific_config"])
        if config.scientific_hash != entry["scientific_config_hash"]:
            raise RuntimeError("scientific config hash mismatch in corpus")
        routes = build_robot_routes(
            solution, config, {robot: directions[robot] for robot in range(4)}
        )
        templates = {
            route.robot_id: build_operation_templates(route, config) for route in routes
        }
        started = time.perf_counter()
        baseline = reference_schedule_from_templates_optimized(
            templates, config, directions=directions
        )
        baseline_runtime = time.perf_counter() - started
        baseline_runtimes.append(baseline_runtime)
        baseline_hash = hashlib.sha256(baseline.canonical_json().encode("utf-8")).hexdigest()
        if baseline.status is not ScheduleStatus.DEADLOCK:
            raise RuntimeError("corpus candidate no longer reproduces baseline DEADLOCK")
        if baseline_hash != entry["baseline"]["canonical_hash"]:
            raise RuntimeError("baseline canonical hash changed during offline replay")
        entry["operation_templates"] = sum(len(rows) for rows in templates.values())
        entry["baseline_runtime"] = baseline_runtime
        entry["replays"] = {}
        found_feasible = False
        previous_cmax = None
        for budget in budgets:
            started = time.perf_counter()
            result = _bounded_dispatch_recovery(
                templates, config, baseline, state_budget=budget
            )
            runtime = time.perf_counter() - started
            result = replace(
                result,
                reference_policy_id=FORMAL_SCOPE_V1.reference_scheduler_policy_id,
                scope_id=FORMAL_SCOPE_V1.scope_id,
                scope_hash=FORMAL_SCOPE_V1.scope_hash,
            )
            if result.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                raise RuntimeError(f"unexpected stress status: {result.status.value}")
            certificate = (
                certify_schedule(solution, result, config, scope=FORMAL_SCOPE_V1)
                if result.feasible
                else None
            )
            if certificate is not None and not certificate.certified:
                raise RuntimeError(
                    f"uncertified recovery {entry['identity']}/{budget}: {certificate.errors}"
                )
            if found_feasible and not result.feasible:
                raise RuntimeError("recovery status is not monotonic in budget")
            if result.feasible:
                found_feasible = True
                if previous_cmax is not None and result.cmax > previous_cmax + config.numeric_epsilon:
                    raise RuntimeError("recovered Cmax worsened at a larger budget")
                previous_cmax = result.cmax
            entry["replays"][str(budget)] = {
                "status": result.status.value,
                "Cmax": result.cmax,
                "expanded_states": result.expanded_states,
                "frontier_exhausted": result.frontier_exhausted,
                "budget_exhausted": result.recovery_exhausted,
                "runtime": runtime,
                "overall_runtime": baseline_runtime + runtime,
                "certified": certificate.certified if certificate else None,
                "schedule_canonical_hash": (
                    hashlib.sha256(result.canonical_json().encode("utf-8")).hexdigest()
                    if result.feasible
                    else None
                ),
            }

    summaries = {}
    for budget in budgets:
        rows = [entry["replays"][str(budget)] for entry in corpus["entries"]]
        recovery_times = [row["runtime"] for row in rows]
        overall_times = [row["overall_runtime"] for row in rows]
        summaries[str(budget)] = {
            "FEASIBLE": sum(row["status"] == ScheduleStatus.FEASIBLE.value for row in rows),
            "DEADLOCK": sum(row["status"] == ScheduleStatus.DEADLOCK.value for row in rows),
            "recovery_p50": _percentile(recovery_times, 0.50),
            "recovery_p95": _percentile(recovery_times, 0.95),
            "overall_p50": _percentile(overall_times, 0.50),
            "overall_p95": _percentile(overall_times, 0.95),
        }
    handover_n100 = [
        entry for entry in corpus["entries"]
        if any(origin["family"] == "handover_heavy" and origin["N"] == 100
               for origin in entry["origins"])
    ]
    corpus["replay_analysis"] = {
        "budgets": list(budgets),
        "baseline_p50": _percentile(baseline_runtimes, 0.50),
        "baseline_p95": _percentile(baseline_runtimes, 0.95),
        "budget_summary": summaries,
        "local_plateau_candidates": detect_local_plateaus(corpus["entries"], budgets),
        "handover_heavy_N100_candidates": len(handover_n100),
        "handover_heavy_N100_certified_feasible": {
            str(budget): sum(
                entry["replays"][str(budget)]["status"] == ScheduleStatus.FEASIBLE.value
                and entry["replays"][str(budget)]["certified"] is True
                for entry in handover_n100
            )
            for budget in budgets
        },
    }
    corpus_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return corpus["replay_analysis"]


def formal_n100_performance(seed: int) -> dict[str, object]:
    provenance = _development_provenance()
    result = run_bounded_sa_oi(
        development_family("handover_heavy", 100),
        ScientificConfig(),
        SearchConfig(max_iterations=100_000, time_limit=5.0),
        seed=seed,
        scope=FORMAL_SCOPE_V1,
        source_provenance=provenance,
    )
    if not result.final_certification or not result.final_certification.certified:
        raise RuntimeError("N100/5s formal performance result is not certified")
    if result.stats.iterations < 1:
        raise RuntimeError("N100/5s formal performance run did not enter search")
    stats = result.stats
    return {
        "stage": "N100_5s_performance",
        "family": "handover_heavy",
        "N": 100,
        "seed": seed,
        "status": result.status.value,
        "certified": result.final_certification.certified,
        "initial_success": result.initialization.status.value == "SUCCESS",
        "iterations": stats.iterations,
        "Nref": stats.nref,
        "requested_budget": stats.requested_budget,
        "actual_runtime": stats.actual_runtime,
        "overshoot": stats.overshoot,
        "init_scheduler_p50": stats.init_scheduler_p50,
        "init_scheduler_p95": stats.init_scheduler_p95,
        "search_scheduler_p50": stats.search_scheduler_p50,
        "search_scheduler_p95": stats.search_scheduler_p95,
        "overall_scheduler_p50": stats.scheduler_p50,
        "overall_scheduler_p95": stats.scheduler_p95,
        "baseline_deadlocks": sum(row["baseline_deadlock"] for row in stats.reference_records),
        "recovered_deadlocks": sum(
            row["source"] == "BOUNDED_DEADLOCK_RECOVERY"
            for row in stats.reference_records
        ),
        "remaining_deadlocks": sum(
            row["baseline_deadlock"] and row["status"] == "DEADLOCK"
            for row in stats.reference_records
        ),
    }


def _rail_x(local: int, total: int) -> float:
    left_count = (total + 1) // 2
    right_count = total // 2
    if local % 2 == 0:
        return 0.7 + 8.0 * (local // 2) / max(1, left_count - 1)
    return 19.2 - 8.0 * (local // 2) / max(1, right_count - 1)


def development_family(name: str, count: int) -> tuple[ParentWeld, ...]:
    parents = []
    per_rail = (count + 1) // 2
    left_count = (per_rail + 1) // 2
    right_count = per_rail // 2
    upper_seen = 0
    lower_seen = 0
    for index in range(count):
        rail_index = index // 2
        upper = index % 2 == 0
        if rail_index % 2 == 0:
            local = rail_index // 2
            x = 0.7 + 8.0 * local / max(1, left_count - 1)
        else:
            local = rail_index // 2
            x = 19.2 - 8.0 * local / max(1, right_count - 1)
        if name == "load_skew":
            upper = index % 4 != 3
            local = upper_seen if upper else lower_seen
            total = count - count // 4 if upper else count // 4
            x = _rail_x(local, total)
            upper_seen += int(upper)
            lower_seen += int(not upper)
            start = (x, 9.0 if upper else 3.0)
            end = (x, 10.0 if upper else 2.0)
        elif name == "spatial_cluster":
            x = 0.7 + (x - 0.7) * 0.25 if rail_index % 2 == 0 else 19.2 - (19.2 - x) * 0.25
            start = (x, 9.0 if upper else 3.0)
            end = (x, 9.25 if upper else 2.75)
        elif name == "handover_heavy":
            local = index // 4
            groups = (count + 3) // 4
            offset = 7.0 * local / max(1, groups - 1)
            x = {
                0: 0.7 + offset,
                1: 19.2 - offset,
                2: 1.3 + offset,
                3: 18.6 - offset,
            }[index % 4]
            start, end = (x, 5.8), (x, 6.2)
        elif name == "interference_stress":
            start = (x, 6.3 if upper else 5.0)
            end = (x, 7.0 if upper else 5.7)
        else:
            raise ValueError(f"unknown development family: {name}")
        parents.append(ParentWeld(f"{name}-{index:03d}", start, end))
    return tuple(parents)


def _profile_solution(parents, config: ScientificConfig) -> dict[str, object]:
    solution = _construct(
        parents,
        _patterns(parents, config),
        config,
        insertion_limit=8,
        strategy=InitializationStrategy.LOAD_FIRST,
    )
    directions = optimize_directions_with_initial_feasibility(solution, config)
    orientation_map = {robot: directions.directions[robot] for robot in range(4)}
    slow_profile = SchedulerProfile()
    slow = reference_schedule_slow(
        solution, config, orientations=orientation_map, profile=slow_profile
    )
    optimized_profile = SchedulerProfile()
    optimized = reference_schedule_optimized(
        solution, config, orientations=orientation_map, profile=optimized_profile
    )
    return {
        "N": len(parents),
        "status": optimized.status.value,
        "scientific_output_equal": slow.canonical_json() == optimized.canonical_json(),
        "operations": optimized_profile.number_of_operations,
        "slow_runtime": slow_profile.total_time,
        "optimized_runtime": optimized_profile.total_time,
        "speedup": slow_profile.total_time / optimized_profile.total_time,
        "slow": slow_profile.as_dict(),
        "optimized": optimized_profile.as_dict(),
    }


def _move_templates(robot_id: int, points, config: ScientificConfig):
    rows = []
    cursor = 0.0
    for index, (start, end) in enumerate(zip(points, points[1:])):
        duration = math.dist(start, end) / config.empty_speed
        rows.append(
            Operation(
                f"stress-R{robot_id}-{index}",
                robot_id,
                OperationKind.MOVE,
                cursor,
                cursor + duration,
                start,
                end,
                index,
            )
        )
        cursor += duration
    return tuple(rows)


def _profile_interference_stress(config: ScientificConfig) -> dict[str, object]:
    templates = {
        0: _move_templates(
            0, ((2.0, 6.0), (2.0, 1.0), (4.0, 6.0), (3.0, 4.0)), config
        ),
        2: _move_templates(
            2, ((1.0, 4.0), (5.0, 4.0), (1.0, 1.0), (2.0, 4.0)), config
        ),
    }
    slow_profile = SchedulerProfile()
    started = time.perf_counter()
    slow = reference_schedule_from_templates_slow(
        templates, config, profile=slow_profile
    )
    slow_profile.total_time = time.perf_counter() - started
    optimized_profile = SchedulerProfile()
    started = time.perf_counter()
    optimized = reference_schedule_from_templates_optimized(
        templates, config, profile=optimized_profile
    )
    optimized_profile.total_time = time.perf_counter() - started
    return {
        "case": "interference_stress_E2",
        "status": optimized.status.value,
        "scientific_output_equal": slow.canonical_json() == optimized.canonical_json(),
        "operations": optimized_profile.number_of_operations,
        "slow_runtime": slow_profile.total_time,
        "optimized_runtime": optimized_profile.total_time,
        "speedup": slow_profile.total_time / optimized_profile.total_time,
        "slow": slow_profile.as_dict(),
        "optimized": optimized_profile.as_dict(),
    }


def _search_summary(parents, config, search_config, seed):
    result = run_bounded_sa_oi(
        parents,
        config,
        search_config,
        seed=seed,
        reference_evaluator=reference_schedule_optimized,
    )
    initial = None if result.initialization.schedule is None else result.initialization.schedule.cmax
    best = None if result.best_schedule is None else result.best_schedule.cmax
    return {
        "N": len(parents),
        "requested_budget": result.stats.requested_budget,
        "actual_runtime": result.stats.actual_runtime,
        "overshoot": result.stats.overshoot,
        "last_reference_start": result.stats.last_reference_start,
        "last_reference_end": result.stats.last_reference_end,
        "iterations": result.stats.iterations,
        "Nref": result.stats.nref,
        "initial_cmax": initial,
        "best_cmax": best,
        "certified": bool(result.final_certification and result.final_certification.certified),
        "init_scheduler_p50": result.stats.init_scheduler_p50,
        "init_scheduler_p95": result.stats.init_scheduler_p95,
        "search_scheduler_p50": result.stats.search_scheduler_p50,
        "search_scheduler_p95": result.stats.search_scheduler_p95,
        "all_scheduler_p50": result.stats.scheduler_p50,
        "all_scheduler_p95": result.stats.scheduler_p95,
    }


def quality_fixture(name: str) -> tuple[tuple[ParentWeld, ...], int, int]:
    cases = {
        "Q1_assignment_trap": (
            (
                ParentWeld("p0", (11.14, 10.0), (12.54, 10.0)),
                ParentWeld("p1", (1.73, 10.0), (3.44, 10.0)),
                ParentWeld("p2", (3.70, 10.0), (4.88, 10.0)),
            ),
            7,
            1,
        ),
        "Q2_route_order_trap": (
            (
                ParentWeld("p0", (7.133, 8.048), (10.143, 11.5)),
                ParentWeld("p1", (10.03, 8.264), (10.365, 8.484)),
                ParentWeld("p2", (13.385, 9.169), (12.862, 9.463)),
                ParentWeld("p3", (12.928, 8.209), (13.724, 8.133)),
            ),
            43,
            8,
        ),
        "Q3_direction_trap": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
            ),
            3,
            20,
        ),
        "Q4_optional_y_split_trap": (
            (ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),),
            0,
            10,
        ),
        "Q5_interference_wait_trap": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
                ParentWeld("d", (2.31, 10.0), (2.59, 10.0)),
            ),
            37,
            12,
        ),
        "Q6_lns_basin_trap": (
            (
                ParentWeld("p0", (1.6988031761970745, 8.06251909281191), (2.507267666737879, 8.024513064718768)),
                ParentWeld("p1", (10.876415580121574, 3.0587891677034196), (11.955019966147066, 3.7709109070254483)),
                ParentWeld("p2", (16.84513432535822, 8.602342619881437), (15.951444969122434, 9.165047149162183)),
                ParentWeld("p3", (6.854207738122211, 1.5241986081119485), (5.40597202291001, 2.096412420693884)),
            ),
            1,
            4,
        ),
    }
    return cases[name]


def _fixed_coordination_cmax(solution, directions, config: ScientificConfig) -> float:
    routes = build_robot_routes(
        solution,
        config,
        {robot: directions[robot] for robot in range(4)},
    )
    templates = {
        route.robot_id: build_operation_templates(route, config) for route in routes
    }
    result = exact_schedule_from_templates(
        templates,
        config,
        directions=directions,
    )
    if result.schedule is None or result.schedule.cmax is None:
        raise RuntimeError("quality fixture fixed coordination is not FEASIBLE")
    return result.schedule.cmax


def _quality_summary(name: str, config: ScientificConfig) -> dict[str, object]:
    parents, seed, iterations = quality_fixture(name)
    exact = solve_exact_micro(parents, config)
    if exact.status is not ExactSolveStatus.OPTIMAL or exact.best_cmax is None:
        raise RuntimeError(f"{name}: exact solve failed: {exact.status.value}")
    search = run_bounded_sa_oi(
        parents,
        config,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=iterations),
        seed=seed,
        reference_evaluator=reference_schedule_optimized,
    )
    initialization = search.initialization
    if (
        initialization.solution is None
        or initialization.directions is None
        or initialization.schedule is None
        or initialization.schedule.cmax is None
    ):
        raise RuntimeError(f"{name}: initialization failed")
    final_gap = micro_gap_decomposition(search, exact, config)
    initial_coord = _fixed_coordination_cmax(
        initialization.solution, initialization.directions, config
    )
    initial_ref = initialization.schedule.cmax
    return {
        "case": name,
        "seed": seed,
        "iterations": search.stats.iterations,
        "C_star": exact.best_cmax,
        "initial_Cref": initial_ref,
        "initial_Ccoord": initial_coord,
        "initial_search_gap": initial_coord - exact.best_cmax,
        "initial_scheduler_gap": initial_ref - initial_coord,
        "final_Cref": final_gap.c_ref,
        "final_Ccoord": final_gap.c_coord,
        "final_search_gap": final_gap.search_gap,
        "final_scheduler_gap": final_gap.scheduler_gap,
        "improved": final_gap.c_ref < initial_ref - 1.0e-9,
        "certified": bool(
            search.final_certification and search.final_certification.certified
        ),
        "initial_wait_operations": sum(
            operation.kind is OperationKind.WAIT
            for operation in initialization.schedule.operations
        ),
        "best_improvement_by_move": {
            move: count
            for move, count in search.stats.best_improvement_by_move.items()
            if count
        },
        "best_improvement_by_family": {
            family: count
            for family, count in search.stats.improvements_by_family.items()
            if count
        },
    }


def _initialization_matrix_row(
    family: str,
    size: int,
    config: ScientificConfig,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    successes = 0
    reference_calls = 0
    winners: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for seed in seeds:
        result = run_bounded_sa_oi(
            development_family(family, size),
            config,
            SearchConfig(max_iterations=0),
            seed=seed,
            reference_evaluator=reference_schedule_optimized,
        )
        reference_calls += result.stats.init_reference_calls
        if result.status.value == "COMPLETED":
            successes += 1
        if result.initialization.winning_strategy:
            winners[result.initialization.winning_strategy] += 1
        for attempt in result.initialization.attempts:
            if attempt.directions is not None and attempt.directions.status.value != "FEASIBLE":
                status_counts[attempt.directions.status.value] += 1
            if attempt.schedule is not None:
                status_counts[attempt.schedule.status.value] += 1
    return {
        "family": family,
        "N": size,
        "seeds": seeds,
        "successes": successes,
        "success_rate": successes / len(seeds),
        "initial_reference_calls": reference_calls,
        "winning_strategies": dict(winners),
        "status_counts": dict(status_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Development reference-scheduler profiler")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--skip-families", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--formal-scope-gate", choices=("all", "calibration", "quality", "smoke"))
    parser.add_argument("--formal-seeds", nargs="+", type=int, default=(20260928, 20260929, 20260930))
    parser.add_argument("--pre-phase3-release", choices=("collect", "replay", "performance"))
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--stress-target", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.pre_phase3_release == "collect":
        corpus = collect_deadlock_stress_corpus(
            arguments.corpus_path,
            seeds=tuple(arguments.formal_seeds),
            target_unique=arguments.stress_target,
        )
        print(json.dumps({
            "stage": "stress_collection",
            "path": str(arguments.corpus_path),
            "raw_deadlock_observations": corpus["raw_deadlock_observations"],
            "unique_entries": corpus["unique_entries"],
            "runs": len(corpus["runs"]),
        }, sort_keys=True))
        return
    if arguments.pre_phase3_release == "replay":
        analysis = replay_deadlock_stress_corpus(arguments.corpus_path)
        print(json.dumps({"stage": "stress_replay", **analysis}, sort_keys=True))
        return
    if arguments.pre_phase3_release == "performance":
        print(json.dumps(formal_n100_performance(arguments.seed), sort_keys=True))
        return
    if arguments.formal_scope_gate:
        formal_scope_gate(arguments.formal_scope_gate, tuple(arguments.formal_seeds))
        return
    config = ScientificConfig()
    scheduler_cases = [
        _profile_solution(synthetic_parents(size), config) for size in (20, 50, 100)
    ]
    scheduler_cases.append(_profile_interference_stress(config))
    search_5s = _search_summary(
        synthetic_parents(100),
        config,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=100_000, time_limit=5.0),
        arguments.seed,
    )
    families = []
    initialization_matrix = []
    if not arguments.skip_families:
        for family in (
            "load_skew",
            "spatial_cluster",
            "handover_heavy",
            "interference_stress",
        ):
            for size in (20, 50, 100):
                summary = _search_summary(
                    development_family(family, size),
                    config,
                    SearchConfig(m=28, kdp=4, kref=1, max_iterations=1),
                    arguments.seed,
                )
                summary["family"] = family
                families.append(summary)
                initialization_matrix.append(
                    _initialization_matrix_row(
                        family,
                        size,
                        config,
                        tuple(arguments.seed + offset for offset in range(5)),
                    )
                )
    quality = []
    if not arguments.skip_quality:
        quality = [
            _quality_summary(name, ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0))
            for name in (
                "Q1_assignment_trap",
                "Q2_route_order_trap",
                "Q3_direction_trap",
                "Q4_optional_y_split_trap",
                "Q5_interference_wait_trap",
                "Q6_lns_basin_trap",
            )
        ]
    print(
        json.dumps(
            {
                "scope_id": "EXACT_Y_SCOPE_CURRENT_SEMANTICS",
                "seed": arguments.seed,
                "scheduler_cases": scheduler_cases,
                "N100_5s_search": search_5s,
                "development_families": families,
                "initialization_matrix": initialization_matrix,
                "search_quality": quality,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
