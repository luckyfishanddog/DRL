from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import random
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
    _limited_discrepancy_dispatch_recovery,
    _limited_discrepancy_dispatch_recovery_slow,
    _limited_discrepancy_dispatch_recovery_optimized,
    _optimized_dispatch_outcome,
    _prepared_dispatch_outcome,
    build_operation_templates,
    build_robot_routes,
    prepare_dispatch_problem,
    reference_schedule_from_templates_optimized,
    reference_schedule_from_templates_slow,
    reference_schedule_optimized,
    reference_schedule_slow,
)
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.provenance import resolve_source_provenance
from mrta_reference.scope import FORMAL_SCOPE_V1, FORMAL_SCOPE_V1_1
from mrta_reference.solution import canonicalize
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

METHOD_INDEPENDENT_SAMPLER_POLICY_ID = "METHOD_INDEPENDENT_DIRECT_SAMPLER_V1"
METHOD_INDEPENDENT_CORPUS_SCHEMA_ID = "F4_METHOD_INDEPENDENT_CALIBRATION_CORPUS_V1"
METHOD_INDEPENDENT_MASTER_SEED = 20260929
METHOD_INDEPENDENT_STRATA = tuple(
    (family, size)
    for family in (
        "load_skew",
        "spatial_cluster",
        "handover_heavy",
        "interference_stress",
    )
    for size in (20, 50, 100)
)
METHOD_INDEPENDENT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "development"
    / "f4_method_independent_calibration_v1.json"
)


def _development_provenance():
    return resolve_source_provenance(
        Path(__file__).resolve().parents[1],
        source_commit="UNVERIFIED_LOCAL_TREE",
        allow_unverified_source=True,
    )


def formal_scope_gate(
    stage: str, seeds: tuple[int, ...], *, scope=FORMAL_SCOPE_V1
) -> None:
    """Development-only freeze evidence; JSON lines on stdout, no dataset files."""
    from mrta_reference.scope import RunScientificIdentity
    from mrta_reference.scheduler import (
        reference_schedule_from_templates_formal, _bounded_dispatch_recovery,
    )
    from mrta_reference.certifier import certify_template_schedule
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
        calibration_budgets = (
            (1, 2, 4, 8, 16, 32)
            if scope is FORMAL_SCOPE_V1_1
            else (16, 32, 64, 128)
        )
        for budget in calibration_budgets:
            for name, templates in fixtures["_manual_oracle_cases"]().items():
                baseline_outcome = _optimized_dispatch_outcome(
                    templates, fast, collect_trace=True
                )
                baseline = baseline_outcome.result
                started = time.perf_counter()
                result = (
                    _limited_discrepancy_dispatch_recovery(
                        templates,
                        fast,
                        baseline_outcome,
                        rollout_budget=budget,
                    )
                    if scope is FORMAL_SCOPE_V1_1
                    else _bounded_dispatch_recovery(
                        templates, fast, baseline, state_budget=budget
                    )
                )
                elapsed = time.perf_counter() - started
                certificate = certify_template_schedule(templates, result, fast) if result.feasible else None
                if result.feasible and not certificate.certified:
                    raise RuntimeError(certificate.errors)
                emit({"stage": "calibration", "case": name, "budget": budget,
                      "baseline": baseline.status.value, "status": result.status.value,
                      "expanded_states": result.expanded_states,
                      "recovery_rollouts": result.recovery_rollouts,
                      "max_discrepancies_used": result.max_discrepancies_used,
                      "Cmax": result.cmax,
                      "certified": bool(certificate and certificate.certified), "runtime": elapsed,
                      "frontier_exhausted": result.frontier_exhausted})
        for name, templates in fixtures["_manual_oracle_cases"]().items():
            start = time.perf_counter()
            result = reference_schedule_from_templates_formal(
                templates, fast, scope=scope
            )
            emit({"stage": "E1-E4", "case": name, "status": result.status.value,
                  "Cmax": result.cmax, "source": result.source, "expanded_states": result.expanded_states,
                  "budget": result.state_budget or result.rollout_budget,
                  "recovery_rollouts": result.recovery_rollouts,
                  "max_discrepancies_used": result.max_discrepancies_used,
                  "runtime": time.perf_counter() - start,
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
                  "recoveries": sum(r["source"] in {
                      "BOUNDED_DEADLOCK_RECOVERY", "LIMITED_DISCREPANCY_RECOVERY"
                  } for r in result.stats.reference_records)})
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
                          "recoveries": sum(r["source"] in {
                              "BOUNDED_DEADLOCK_RECOVERY", "LIMITED_DISCREPANCY_RECOVERY"
                          } for r in s.reference_records),
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


def _entry_dispatch_problem(entry):
    solution = _solution_from_payload(entry["canonical_solution"])
    config = _config_from_mapping(entry["scientific_config"])
    directions = tuple(tuple(row) for row in entry["directions"])
    routes = build_robot_routes(
        solution, config, {robot: directions[robot] for robot in range(4)}
    )
    templates = {
        route.robot_id: build_operation_templates(route, config) for route in routes
    }
    return solution, config, directions, templates


def performance_closure_profile(corpus_path: Path) -> dict[str, object]:
    """Profile the frozen known recovery and three deterministic N100 cases."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    known = "328926b7612633797dbe0ae62b62964bbf0a6765b1ef0d6147305b07a0ec6511"
    known_entry = next(row for row in corpus["entries"] if row["identity"] == known)
    n100 = [
        row
        for row in corpus["entries"]
        if row["identity"] != known
        and any(origin["N"] == 100 for origin in row["origins"])
    ][:3]
    rows = []
    for entry in (known_entry, *n100):
        _, config, directions, templates = _entry_dispatch_problem(entry)
        profile = SchedulerProfile()
        started = time.perf_counter()
        prepared = prepare_dispatch_problem(
            templates, config, directions=directions, profile=profile
        )
        baseline = _prepared_dispatch_outcome(
            prepared, profile=profile, collect_trace=True
        )
        result = _limited_discrepancy_dispatch_recovery_optimized(
            prepared,
            baseline,
            rollout_budget=FORMAL_SCOPE_V1_1.deadlock_rollout_budget,
            profile=profile,
        )
        elapsed = time.perf_counter() - started
        metrics = profile.as_dict()
        timed = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, float) and value > 0.0
        }
        rows.append({
            "identity": entry["identity"],
            "status": result.status.value,
            "runtime": elapsed,
            "recovery_rollouts": result.recovery_rollouts,
            "profile": metrics,
            "time_percent": {
                key: 100.0 * value / elapsed for key, value in timed.items()
            },
            "cache_entries": {
                "forbidden_intervals": len(prepared.forbidden_interval_cache),
                "conflicts": len(prepared.conflict_cache),
                "ESS": len(prepared.ess_cache),
                "fixed_index_snapshots": len(prepared.fixed_index_snapshots),
            },
        })
    return {"stage": "v1_1_performance_profile", "cases": rows}


def performance_closure_differential(
    corpus_path: Path, *, rollout_budget: int = 32
) -> dict[str, object]:
    """Run the 31-state slow/fast oracle differential without mutating corpus."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    rows = []
    for entry in corpus["entries"]:
        solution, config, directions, templates = _entry_dispatch_problem(entry)
        started = time.perf_counter()
        slow_baseline = _optimized_dispatch_outcome(
            templates, config, directions=directions, collect_trace=True
        )
        slow = _limited_discrepancy_dispatch_recovery_slow(
            templates,
            config,
            slow_baseline,
            rollout_budget=rollout_budget,
        )
        slow_runtime = time.perf_counter() - started

        profile = SchedulerProfile()
        started = time.perf_counter()
        prepared = prepare_dispatch_problem(
            templates, config, directions=directions, profile=profile
        )
        fast_baseline = _prepared_dispatch_outcome(
            prepared, profile=profile, collect_trace=True
        )
        fast = _limited_discrepancy_dispatch_recovery_optimized(
            prepared,
            fast_baseline,
            rollout_budget=rollout_budget,
            profile=profile,
        )
        fast_runtime = time.perf_counter() - started
        if slow_baseline.result != fast_baseline.result or slow_baseline.trace != fast_baseline.trace:
            raise RuntimeError(f"baseline/trace differential: {entry['identity']}")
        if slow != fast:
            raise RuntimeError(f"recovery scientific differential: {entry['identity']}")
        formal = replace(
            fast,
            reference_policy_id=FORMAL_SCOPE_V1_1.reference_scheduler_policy_id,
            scope_id=FORMAL_SCOPE_V1_1.scope_id,
            scope_hash=FORMAL_SCOPE_V1_1.scope_hash,
        )
        certificate = (
            certify_schedule(solution, formal, config, scope=FORMAL_SCOPE_V1_1)
            if formal.feasible
            else None
        )
        if certificate is not None and not certificate.certified:
            raise RuntimeError(
                f"uncertified optimized result {entry['identity']}: {certificate.errors}"
            )
        rows.append({
            "identity": entry["identity"],
            "status": fast.status.value,
            "Cmax": fast.cmax,
            "recovery_rollouts": fast.recovery_rollouts,
            "slow_runtime": slow_runtime,
            "optimized_runtime": fast_runtime,
            "speedup": slow_runtime / fast_runtime,
            "certified": certificate.certified if certificate else None,
            "profile": profile.as_dict(),
            "cache_entries": {
                "forbidden_intervals": len(prepared.forbidden_interval_cache),
                "conflicts": len(prepared.conflict_cache),
                "ESS": len(prepared.ess_cache),
                "fixed_index_snapshots": len(prepared.fixed_index_snapshots),
            },
        })
        print(json.dumps({"stage": "v1_1_differential_case", **rows[-1]}, sort_keys=True), flush=True)
    slow_times = [row["slow_runtime"] for row in rows]
    fast_times = [row["optimized_runtime"] for row in rows]
    speedups = [row["speedup"] for row in rows]
    slow_p50, slow_p95 = _percentile(slow_times, 0.5), _percentile(slow_times, 0.95)
    fast_p50, fast_p95 = _percentile(fast_times, 0.5), _percentile(fast_times, 0.95)
    return {
        "stage": "v1_1_performance_differential",
        "entries": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "slow_p50": slow_p50,
        "slow_p95": slow_p95,
        "optimized_p50": fast_p50,
        "optimized_p95": fast_p95,
        "median_case_speedup": _percentile(speedups, 0.5),
        "p95_case_speedup": _percentile(speedups, 0.95),
        "p50_distribution_speedup": slow_p50 / fast_p50,
        "p95_distribution_speedup": slow_p95 / fast_p95,
        "scientific_differentials": 0,
        "certified_feasible": sum(row["certified"] is True for row in rows),
    }


def performance_closure_marginal(corpus_path: Path) -> dict[str, object]:
    """Final B32/B64 diagnostic; invoke only after optimized B32 is frozen."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    rows = []
    for entry in corpus["entries"]:
        solution, config, directions, templates = _entry_dispatch_problem(entry)
        by_budget = {}
        for budget in (32, 64):
            started = time.perf_counter()
            prepared = prepare_dispatch_problem(templates, config, directions=directions)
            baseline = _prepared_dispatch_outcome(prepared, collect_trace=True)
            result = _limited_discrepancy_dispatch_recovery_optimized(
                prepared, baseline, rollout_budget=budget
            )
            runtime = time.perf_counter() - started
            formal = replace(
                result,
                reference_policy_id=FORMAL_SCOPE_V1_1.reference_scheduler_policy_id,
                scope_id=FORMAL_SCOPE_V1_1.scope_id,
                scope_hash=FORMAL_SCOPE_V1_1.scope_hash,
            )
            certificate = (
                certify_schedule(solution, formal, config, scope=FORMAL_SCOPE_V1_1)
                if formal.feasible
                else None
            )
            if certificate is not None and not certificate.certified:
                raise RuntimeError(f"uncertified B{budget}: {entry['identity']}")
            by_budget[budget] = {
                "status": result.status.value,
                "Cmax": result.cmax,
                "runtime": runtime,
                "rollouts": result.recovery_rollouts,
                "certified": certificate.certified if certificate else None,
            }
        if by_budget[32]["status"] == ScheduleStatus.FEASIBLE.value:
            if by_budget[64]["status"] != ScheduleStatus.FEASIBLE.value:
                raise RuntimeError(f"B64 regressed status: {entry['identity']}")
            if by_budget[64]["Cmax"] > by_budget[32]["Cmax"] + config.numeric_epsilon:
                raise RuntimeError(f"B64 worsened Cmax: {entry['identity']}")
        rows.append({"identity": entry["identity"], "budgets": by_budget})
        print(json.dumps({"stage": "v1_1_marginal_case", **rows[-1]}, sort_keys=True), flush=True)
    summaries = {}
    for budget in (32, 64):
        runtimes = [row["budgets"][budget]["runtime"] for row in rows]
        summaries[str(budget)] = {
            "FEASIBLE": sum(
                row["budgets"][budget]["status"] == ScheduleStatus.FEASIBLE.value
                for row in rows
            ),
            "DEADLOCK": sum(
                row["budgets"][budget]["status"] == ScheduleStatus.DEADLOCK.value
                for row in rows
            ),
            "runtime_p50": _percentile(runtimes, 0.5),
            "runtime_p95": _percentile(runtimes, 0.95),
            "rollouts": sum(row["budgets"][budget]["rollouts"] for row in rows),
        }
    added = summaries["64"]["FEASIBLE"] - summaries["32"]["FEASIBLE"]
    return {
        "stage": "v1_1_b32_b64_marginal",
        "entries": len(rows),
        "budgets": summaries,
        "added_certified_recoveries": added,
        "release_rule": "A" if added <= 1 else "B",
    }


def replay_historical_corpus_final(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> dict[str, object]:
    """Cross-distribution B32/B64/B128 diagnostic after budget selection."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    summaries = {
        budget: {"FEASIBLE": 0, "DEADLOCK": 0, "certified": 0, "runtimes": []}
        for budget in (32, 64, 128)
    }
    for entry in corpus["entries"]:
        solution, config, directions, templates = _entry_dispatch_problem(entry)
        previous = None
        for budget in (32, 64, 128):
            started = time.perf_counter()
            prepared = prepare_dispatch_problem(templates, config, directions=directions)
            baseline = _prepared_dispatch_outcome(prepared, collect_trace=True)
            result = _limited_discrepancy_dispatch_recovery_optimized(
                prepared, baseline, rollout_budget=budget
            )
            elapsed = time.perf_counter() - started
            certificate = certify_schedule(solution, result, config) if result.feasible else None
            if certificate is not None and not certificate.certified:
                raise RuntimeError(f"uncertified historical B{budget}: {entry['identity']}")
            summaries[budget][result.status.value] += 1
            summaries[budget]["certified"] += int(bool(certificate and certificate.certified))
            summaries[budget]["runtimes"].append(elapsed)
            if previous is not None and previous.feasible:
                if not result.feasible:
                    raise RuntimeError(f"historical B{budget} status regression")
                if result.cmax > previous.cmax + config.numeric_epsilon:
                    raise RuntimeError(f"historical B{budget} Cmax regression")
            previous = result
    return {
        "entries": len(corpus["entries"]),
        "selected_budget": 32,
        "budgets": {
            str(budget): {
                "FEASIBLE": row["FEASIBLE"],
                "DEADLOCK": row["DEADLOCK"],
                "certified": row["certified"],
                "runtime_p50": _percentile(row["runtimes"], 0.50),
                "runtime_p95": _percentile(row["runtimes"], 0.95),
            }
            for budget, row in summaries.items()
        },
        "monotonicity_violations": 0,
        "certification_failures": 0,
    }


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


def replay_v1_1_deadlock_stress_corpus(
    corpus_path: Path,
    *,
    budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[str, object]:
    """Replay the frozen V1 corpus with the V1.1 complete-rollout policy."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus["schema_id"] != "F4_DEADLOCK_STRESS_CORPUS_V1":
        raise RuntimeError("unsupported stress corpus schema")
    baseline_runtimes: list[float] = []
    known_identity = "328926b7612633797dbe0ae62b62964bbf0a6765b1ef0d6147305b07a0ec6511"
    known_recovered_within_16 = False
    for entry in corpus["entries"]:
        solution = _solution_from_payload(entry["canonical_solution"])
        directions = tuple(tuple(row) for row in entry["directions"])
        config = _config_from_mapping(entry["scientific_config"])
        routes = build_robot_routes(
            solution, config, {robot: directions[robot] for robot in range(4)}
        )
        templates = {
            route.robot_id: build_operation_templates(route, config) for route in routes
        }
        started = time.perf_counter()
        baseline_outcome = _optimized_dispatch_outcome(
            templates, config, directions=directions, collect_trace=True
        )
        baseline_runtime = time.perf_counter() - started
        baseline = baseline_outcome.result
        baseline_runtimes.append(baseline_runtime)
        if baseline.status is not ScheduleStatus.DEADLOCK:
            raise RuntimeError("corpus candidate no longer reproduces baseline DEADLOCK")
        baseline_hash = hashlib.sha256(
            baseline.canonical_json().encode("utf-8")
        ).hexdigest()
        if baseline_hash != entry["baseline"]["canonical_hash"]:
            raise RuntimeError("V1.1 replay changed the baseline scheduler output")
        entry["v1_1_baseline_runtime"] = baseline_runtime
        entry["v1_1_branch_points"] = len(baseline_outcome.trace.branch_points)
        entry["v1_1_replays"] = {}
        found_feasible = False
        previous_cmax = None
        for budget in budgets:
            started = time.perf_counter()
            result = _limited_discrepancy_dispatch_recovery(
                templates,
                config,
                baseline_outcome,
                rollout_budget=budget,
            )
            recovery_runtime = time.perf_counter() - started
            result = replace(
                result,
                reference_policy_id=FORMAL_SCOPE_V1_1.reference_scheduler_policy_id,
                scope_id=FORMAL_SCOPE_V1_1.scope_id,
                scope_hash=FORMAL_SCOPE_V1_1.scope_hash,
                rollout_budget=budget,
            )
            if result.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
                raise RuntimeError(f"unexpected V1.1 stress status: {result.status.value}")
            certificate = (
                certify_schedule(solution, result, config, scope=FORMAL_SCOPE_V1_1)
                if result.feasible
                else None
            )
            if certificate is not None and not certificate.certified:
                raise RuntimeError(
                    f"uncertified V1.1 recovery {entry['identity']}/{budget}: "
                    f"{certificate.errors}"
                )
            if found_feasible and not result.feasible:
                raise RuntimeError("V1.1 recovery status is not monotonic in budget")
            if result.feasible:
                found_feasible = True
                if (
                    previous_cmax is not None
                    and result.cmax > previous_cmax + config.numeric_epsilon
                ):
                    raise RuntimeError("V1.1 recovered Cmax worsened at a larger budget")
                previous_cmax = result.cmax
            if entry["identity"] == known_identity and budget <= 16 and result.feasible:
                known_recovered_within_16 = True
            entry["v1_1_replays"][str(budget)] = {
                "status": result.status.value,
                "Cmax": result.cmax,
                "recovery_rollouts": result.recovery_rollouts,
                "rollout_budget": budget,
                "max_discrepancies_used": result.max_discrepancies_used,
                "branch_points_considered": result.branch_points_considered,
                "frontier_exhausted": result.frontier_exhausted,
                "budget_exhausted": result.recovery_exhausted,
                "runtime": recovery_runtime,
                "overall_runtime": baseline_runtime + recovery_runtime,
                "certified": certificate.certified if certificate else None,
                "schedule_canonical_hash": (
                    hashlib.sha256(result.canonical_json().encode("utf-8")).hexdigest()
                    if result.feasible
                    else None
                ),
            }
    if not known_recovered_within_16:
        raise RuntimeError("known 2048-prefix N100 case did not recover within 16 rollouts")

    summaries = {}
    for budget in budgets:
        rows = [entry["v1_1_replays"][str(budget)] for entry in corpus["entries"]]
        recovery_times = [row["runtime"] for row in rows]
        overall_times = [row["overall_runtime"] for row in rows]
        rollout_counts = [row["recovery_rollouts"] for row in rows]
        summaries[str(budget)] = {
            "FEASIBLE": sum(row["status"] == ScheduleStatus.FEASIBLE.value for row in rows),
            "DEADLOCK": sum(row["status"] == ScheduleStatus.DEADLOCK.value for row in rows),
            "certified_feasible": sum(row["certified"] is True for row in rows),
            "recovery_p50": _percentile(recovery_times, 0.50),
            "recovery_p95": _percentile(recovery_times, 0.95),
            "overall_p50": _percentile(overall_times, 0.50),
            "overall_p95": _percentile(overall_times, 0.95),
            "rollouts_mean": sum(rollout_counts) / len(rollout_counts),
            "rollouts_median": _percentile(rollout_counts, 0.50),
            "max_discrepancies": max(row["max_discrepancies_used"] for row in rows),
        }
    analysis = {
        "scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "selected_rollout_budget": FORMAL_SCOPE_V1_1.deadlock_rollout_budget,
        "selection_reason": (
            "B16 recovered the known case, but B32 added eight certified corpus "
            "recoveries; freeze the largest calibrated complete-rollout budget"
        ),
        "budgets": list(budgets),
        "entries": len(corpus["entries"]),
        "baseline_p50": _percentile(baseline_runtimes, 0.50),
        "baseline_p95": _percentile(baseline_runtimes, 0.95),
        "budget_summary": summaries,
        "known_2048_prefix_case_recovered_within_16": known_recovered_within_16,
    }
    corpus["v1_1_replay_analysis"] = analysis
    corpus_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return analysis


def formal_n100_performance(
    seed: int, *, scope=FORMAL_SCOPE_V1, time_limit: float = 5.0
) -> dict[str, object]:
    provenance = _development_provenance()
    result = run_bounded_sa_oi(
        development_family("handover_heavy", 100),
        ScientificConfig(),
        SearchConfig(max_iterations=100_000, time_limit=time_limit),
        seed=seed,
        scope=scope,
        source_provenance=provenance,
    )
    if not result.final_certification or not result.final_certification.certified:
        raise RuntimeError("N100 formal performance result is not certified")
    if result.stats.iterations < 1:
        raise RuntimeError("N100 formal performance run did not enter search")
    stats = result.stats
    checkpoints = stats.anytime((1.0, 5.0, 30.0))
    return {
        "stage": "N100_performance",
        "family": "handover_heavy",
        "N": 100,
        "seed": seed,
        "status": result.status.value,
        "certified": result.final_certification.certified,
        "initial_success": result.initialization.status.value == "SUCCESS",
        "initial_Cmax": result.initialization.schedule.cmax,
        "best_Cmax": result.best_schedule.cmax,
        "Cmax_at_1": checkpoints[1.0]["cmax"],
        "Cmax_at_5": checkpoints[5.0]["cmax"],
        "Cmax_at_30": checkpoints[30.0]["cmax"],
        "iterations": stats.iterations,
        "Nref": stats.nref,
        "C4_feasible_rate": stats.n_feasible / stats.nref if stats.nref else None,
        "requested_budget": stats.requested_budget,
        "actual_runtime": stats.actual_runtime,
        "overshoot": stats.overshoot,
        "init_scheduler_p50": stats.init_scheduler_p50,
        "init_scheduler_p95": stats.init_scheduler_p95,
        "search_scheduler_p50": stats.search_scheduler_p50,
        "search_scheduler_p95": stats.search_scheduler_p95,
        "overall_scheduler_p50": stats.scheduler_p50,
        "overall_scheduler_p95": stats.scheduler_p95,
        "repair_time": stats.repair_time,
        "scheduler_time": stats.reference_scheduler_time,
        "certifier_time": stats.certifier_time,
        "candidate_generation_time": stats.candidate_generation_time,
        "baseline_deadlocks": sum(row["baseline_deadlock"] for row in stats.reference_records),
        "recovered_deadlocks": sum(
            row["source"] in {
                "BOUNDED_DEADLOCK_RECOVERY", "LIMITED_DISCREPANCY_RECOVERY"
            }
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


def _direct_sampling_seed(
    master_seed: int, family: str, size: int, sample_ordinal: int
) -> int:
    payload = json.dumps(
        {
            "family": family,
            "master_seed": master_seed,
            "policy": METHOD_INDEPENDENT_SAMPLER_POLICY_ID,
            "sample_ordinal": sample_ordinal,
            "size": size,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _sample_formal_pattern(
    parent: ParentWeld, config: ScientificConfig, rng: random.Random
) -> SplitPattern:
    """Sample pattern kind first so Y candidate multiplicity cannot bias the kind."""
    y_patterns = tuple(
        pattern
        for pattern in generate_y_split_patterns(parent, config)
        if all(
            any(robot_is_eligible(block, robot, config) for robot in range(4))
            for block in blocks_for_pattern(parent, pattern, config)
        )
    )
    whole_eligible = bool(whole_eligible_rails(parent.start, parent.end, config))
    if not whole_eligible:
        if not y_patterns:
            raise ValueError(f"{parent.parent_id}: mandatory Y has no legal candidate")
        return rng.choice(y_patterns)
    if not y_patterns or rng.randrange(2) == 0:
        return SplitPattern(parent.parent_id, SplitKind.WHOLE)
    return rng.choice(y_patterns)


def sample_method_independent_solution(
    parents: tuple[ParentWeld, ...],
    config: ScientificConfig,
    *,
    master_seed: int,
    family: str,
    size: int,
    sample_ordinal: int,
) -> tuple[CanonicalSolution, tuple[tuple[int, ...], ...], dict[str, object]]:
    """Directly sample the formal structural domain without a search heuristic."""
    sampling_seed = _direct_sampling_seed(master_seed, family, size, sample_ordinal)
    rng = random.Random(sampling_seed)
    ordered_parents = tuple(sorted(parents, key=lambda parent: parent.parent_id))
    for structural_draw in range(1, 4097):
        patterns = tuple(
            _sample_formal_pattern(parent, config, rng) for parent in ordered_parents
        )
        blocks = sorted(
            (
                block
                for parent, pattern in zip(ordered_parents, patterns)
                for block in blocks_for_pattern(parent, pattern, config)
            ),
            key=lambda block: block.block_id,
        )
        assigned = [[] for _ in range(4)]
        for block in blocks:
            eligible = tuple(
                robot for robot in range(4) if robot_is_eligible(block, robot, config)
            )
            if not eligible:
                raise ValueError(f"{block.block_id}: no formally eligible robot")
            assigned[rng.choice(eligible)].append(block.block_id)
        for block_ids in assigned:
            block_ids.sort()
            rng.shuffle(block_ids)
        routes = tuple(Route(robot, tuple(assigned[robot])) for robot in range(4))
        try:
            solution = canonicalize(ordered_parents, patterns, routes, config)
        except ValueError:
            # Mandatory-Y children that become adjacent on one robot are not a
            # canonical formal solution. Rejection sampling preserves the
            # direct uniform draws while returning only structural-domain rows.
            continue
        directions = tuple(
            tuple(rng.randrange(2) for _ in route.block_ids)
            for route in solution.routes
        )
        metadata = {
            "sampler_policy_id": METHOD_INDEPENDENT_SAMPLER_POLICY_ID,
            "master_seed": master_seed,
            "stratum": {"family": family, "N": size},
            "sample_ordinal": sample_ordinal,
            "sampling_seed": sampling_seed,
            "structural_draws": structural_draw,
        }
        return solution, directions, metadata
    raise RuntimeError("direct sampler exceeded 4096 deterministic structural draws")


def _direct_state_exact_key(
    solution: CanonicalSolution,
    directions: tuple[tuple[int, ...], ...],
    config: ScientificConfig,
) -> str:
    return json.dumps(
        {
            "canonical_solution_hash": solution.canonical_hash,
            "directions": directions,
            "scientific_config_hash": config.scientific_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _direct_corpus_entry(
    solution: CanonicalSolution,
    directions: tuple[tuple[int, ...], ...],
    config: ScientificConfig,
    metadata: dict[str, object],
    baseline,
) -> dict[str, object]:
    exact_key = _direct_state_exact_key(solution, directions, config)
    return {
        "identity": hashlib.sha256(exact_key.encode("utf-8")).hexdigest(),
        "canonical_solution_hash": solution.canonical_hash,
        "canonical_solution": solution.canonical_payload(),
        "directions": [list(row) for row in directions],
        "scientific_config_hash": config.scientific_hash,
        "scientific_config": json.loads(json.dumps(asdict(config), allow_nan=False)),
        **metadata,
        "baseline": {
            "status": baseline.status.value,
            "source": baseline.source,
            "diagnostics": list(baseline.diagnostics),
            "wait_for_graph": [
                [robot, list(blockers)] for robot, blockers in baseline.wait_for_graph
            ],
            "canonical_schedule_hash": hashlib.sha256(
                baseline.canonical_json().encode("utf-8")
            ).hexdigest(),
        },
    }


def _baseline_status_is_collectible(status: ScheduleStatus) -> bool:
    return status is ScheduleStatus.DEADLOCK


def build_method_independent_corpus(
    *, master_seed: int = METHOD_INDEPENDENT_MASTER_SEED
) -> dict[str, object]:
    """Execute the predeclared two-stage collection policy entirely in memory."""
    config = ScientificConfig()
    accepted: list[dict[str, object]] = []
    exact_seen: set[str] = set()
    stats = {
        f"{family}/N{size}": {
            "family": family,
            "N": size,
            "attempts": 0,
            "accepted_unique_deadlocks": 0,
            "duplicate_deadlocks": 0,
            "status_counts": {},
        }
        for family, size in METHOD_INDEPENDENT_STRATA
    }

    def execute_stage(start: int, stop: int, cap: int) -> None:
        for family, size in METHOD_INDEPENDENT_STRATA:
            parents = development_family(family, size)
            row = stats[f"{family}/N{size}"]
            counts = Counter(row["status_counts"])
            for ordinal in range(start, stop):
                solution, directions, metadata = sample_method_independent_solution(
                    parents,
                    config,
                    master_seed=master_seed,
                    family=family,
                    size=size,
                    sample_ordinal=ordinal,
                )
                orientation_map = {
                    robot: directions[robot] for robot in range(4)
                }
                baseline = reference_schedule_optimized(
                    solution, config, orientations=orientation_map
                )
                row["attempts"] += 1
                counts[baseline.status.value] += 1
                if not _baseline_status_is_collectible(baseline.status):
                    continue
                exact_key = _direct_state_exact_key(solution, directions, config)
                if exact_key in exact_seen:
                    row["duplicate_deadlocks"] += 1
                    continue
                exact_seen.add(exact_key)
                if row["accepted_unique_deadlocks"] >= cap:
                    continue
                entry = _direct_corpus_entry(
                    solution, directions, config, metadata, baseline
                )
                # Digest is an external label only; exact-key membership above
                # prevents a theoretical SHA collision from merging states.
                entry["collection_index"] = len(accepted)
                accepted.append(entry)
                row["accepted_unique_deadlocks"] += 1
            row["status_counts"] = dict(sorted(counts.items()))

    execute_stage(0, 256, 8)
    stage = "A"
    if len(accepted) < 48:
        execute_stage(256, 1024, 12)
        stage = "B"
    corpus = {
        "schema_id": METHOD_INDEPENDENT_CORPUS_SCHEMA_ID,
        "development_only": True,
        "future_validation_test_ood_used": False,
        "method_source": "DIRECT_FORMAL_DOMAIN_SAMPLING",
        "sampler_policy_id": METHOD_INDEPENDENT_SAMPLER_POLICY_ID,
        "master_seed": master_seed,
        "scope_id_at_collection": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash_at_collection": FORMAL_SCOPE_V1_1.scope_hash,
        "scientific_config_hash": config.scientific_hash,
        "collection_policy": {
            "strata": [
                {"family": family, "N": size}
                for family, size in METHOD_INDEPENDENT_STRATA
            ],
            "stage_a_attempts_per_stratum": 256,
            "stage_a_max_unique_deadlocks_per_stratum": 8,
            "stage_a_stop_total": 48,
            "stage_b_total_attempts_per_stratum": 1024,
            "stage_b_max_unique_deadlocks_per_stratum": 12,
            "completed_stage": stage,
        },
        "strata": [stats[f"{family}/N{size}"] for family, size in METHOD_INDEPENDENT_STRATA],
        "unique_baseline_deadlocks": len(accepted),
        "entries": accepted,
    }
    # Prove that the in-memory payload is JSON round-trip exact before a caller
    # is allowed to freeze it on disk.
    encoded = json.dumps(corpus, sort_keys=True, allow_nan=False)
    if json.loads(encoded) != corpus:
        raise RuntimeError("method-independent corpus JSON round-trip failed")
    return corpus


def write_method_independent_corpus(
    output_path: Path = METHOD_INDEPENDENT_CORPUS_PATH,
    *,
    master_seed: int = METHOD_INDEPENDENT_MASTER_SEED,
) -> dict[str, object]:
    corpus = build_method_independent_corpus(master_seed=master_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if json.loads(output_path.read_text(encoding="utf-8")) != corpus:
        raise RuntimeError("frozen method-independent corpus did not round-trip")
    return corpus


def select_final_deadlock_budget(
    recovery_counts: dict[int, int], n100_p95: dict[int, float | None]
) -> int | None:
    """Apply the predeclared 90%-of-B128 coverage and eight-second gate."""
    r128 = recovery_counts[128]
    for budget in (32, 64, 128):
        coverage = 1.0 if r128 == 0 else recovery_counts[budget] / r128
        p95 = n100_p95.get(budget)
        if coverage >= 0.90 and p95 is not None and p95 <= 8.0:
            return budget
    return None


def calibrate_method_independent_corpus(
    corpus_path: Path = METHOD_INDEPENDENT_CORPUS_PATH,
    *,
    emit_cases: bool = False,
) -> dict[str, object]:
    """Replay the already-frozen corpus at exactly B32/B64/B128."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus.get("schema_id") != METHOD_INDEPENDENT_CORPUS_SCHEMA_ID:
        raise ValueError("unexpected method-independent corpus schema")
    if corpus.get("unique_baseline_deadlocks", 0) < 31:
        raise RuntimeError("INSUFFICIENT_METHOD_INDEPENDENT_DEADLOCK_EVIDENCE")
    rows = [
        {
            "identity": entry["identity"],
            "family": entry["stratum"]["family"],
            "N": entry["stratum"]["N"],
            "sample_ordinal": entry["sample_ordinal"],
            "budgets": {},
        }
        for entry in corpus["entries"]
    ]
    for budget in (32, 64, 128):
        for entry, row in zip(corpus["entries"], rows):
            solution, config, directions, templates = _entry_dispatch_problem(entry)
            profile = SchedulerProfile()
            started = time.perf_counter()
            prepared = prepare_dispatch_problem(
                templates, config, directions=directions, profile=profile
            )
            baseline = _prepared_dispatch_outcome(
                prepared, profile=profile, collect_trace=True
            )
            if baseline.result.status is not ScheduleStatus.DEADLOCK:
                raise RuntimeError(
                    f"frozen baseline no longer DEADLOCK: {entry['identity']}"
                )
            baseline_hash = hashlib.sha256(
                baseline.result.canonical_json().encode("utf-8")
            ).hexdigest()
            if baseline_hash != entry["baseline"]["canonical_schedule_hash"]:
                raise RuntimeError(
                    f"frozen baseline schedule changed: {entry['identity']}"
                )
            result = _limited_discrepancy_dispatch_recovery_optimized(
                prepared, baseline, rollout_budget=budget, profile=profile
            )
            runtime = time.perf_counter() - started
            if result.rollout_budget != budget or result.recovery_rollouts > budget:
                raise RuntimeError(
                    f"B{budget} rollout cap violated: {entry['identity']}"
                )
            certificate = (
                certify_schedule(solution, result, config) if result.feasible else None
            )
            if certificate is not None and not certificate.certified:
                raise RuntimeError(
                    f"uncertified B{budget}: {entry['identity']}: {certificate.errors}"
                )
            row["budgets"][str(budget)] = {
                "status": result.status.value,
                "Cmax": result.cmax,
                "runtime": runtime,
                "recovery_rollouts": result.recovery_rollouts,
                "max_discrepancies_used": result.max_discrepancies_used,
                "branch_points_considered": result.branch_points_considered,
                "frontier_exhausted": result.frontier_exhausted,
                "certified": certificate.certified if certificate else None,
                "canonical_schedule_hash": hashlib.sha256(
                    result.canonical_json().encode("utf-8")
                ).hexdigest(),
            }
            if emit_cases:
                print(
                    json.dumps(
                        {
                            "stage": "final_f4_calibration_case",
                            "identity": row["identity"],
                            "family": row["family"],
                            "N": row["N"],
                            "budget": budget,
                            **row["budgets"][str(budget)],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    for entry, row in zip(corpus["entries"], rows):
        config = _config_from_mapping(entry["scientific_config"])
        for smaller, larger in ((32, 64), (64, 128)):
            left, right = row["budgets"][str(smaller)], row["budgets"][str(larger)]
            if left["status"] == ScheduleStatus.FEASIBLE.value:
                if right["status"] != ScheduleStatus.FEASIBLE.value:
                    raise RuntimeError(
                        f"B{larger} status regression: {entry['identity']}"
                    )
                if right["Cmax"] > left["Cmax"] + config.numeric_epsilon:
                    raise RuntimeError(
                        f"B{larger} Cmax regression: {entry['identity']}"
                    )
    counts = {
        budget: sum(
            row["budgets"][str(budget)]["status"] == ScheduleStatus.FEASIBLE.value
            for row in rows
        )
        for budget in (32, 64, 128)
    }
    runtime = {}
    n100_p95 = {}
    for budget in (32, 64, 128):
        all_times = [row["budgets"][str(budget)]["runtime"] for row in rows]
        n100_times = [
            row["budgets"][str(budget)]["runtime"]
            for row in rows
            if row["N"] == 100
        ]
        rollouts = [
            row["budgets"][str(budget)]["recovery_rollouts"] for row in rows
        ]
        n100_p95[budget] = _percentile(n100_times, 0.95)
        runtime[str(budget)] = {
            "all_p50": _percentile(all_times, 0.50),
            "all_p95": _percentile(all_times, 0.95),
            "N100_count": len(n100_times),
            "N100_p50": _percentile(n100_times, 0.50),
            "N100_p95": n100_p95[budget],
            "N100_mean": sum(n100_times) / len(n100_times) if n100_times else None,
            "N100_max": max(n100_times) if n100_times else None,
            "rollout_mean": sum(rollouts) / len(rollouts),
            "rollout_median": _percentile(rollouts, 0.50),
        }
    selected = select_final_deadlock_budget(counts, n100_p95)
    return {
        "schema_id": "FINAL_F4_DEVELOPMENT_CALIBRATION_RESULT_V1",
        "development_only": True,
        "corpus_path": str(corpus_path),
        "corpus_entries": len(rows),
        "recovery_counts": {str(key): value for key, value in counts.items()},
        "coverage_of_B128": {
            str(budget): (1.0 if counts[128] == 0 else counts[budget] / counts[128])
            for budget in (32, 64, 128)
        },
        "runtime": runtime,
        "monotonicity_violations": 0,
        "certification_failures": 0,
        "selected_budget": selected,
        "selection_rule": "SMALLEST_B_WITH_RECOVERY_AT_LEAST_90_PERCENT_OF_B128_AND_N100_P95_LE_8S",
        "rows": rows,
    }


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
    parser.add_argument(
        "--formal-scope-v1-1",
        choices=("all", "calibration", "quality", "smoke", "replay", "performance"),
    )
    parser.add_argument("--formal-seeds", nargs="+", type=int, default=(20260928, 20260929, 20260930))
    parser.add_argument("--pre-phase3-release", choices=("collect", "replay", "performance"))
    parser.add_argument(
        "--performance-closure",
        choices=("profile", "differential", "marginal", "n100"),
    )
    parser.add_argument(
        "--final-f4-release",
        choices=("collect", "calibrate"),
    )
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--method-corpus-path", type=Path, default=METHOD_INDEPENDENT_CORPUS_PATH
    )
    parser.add_argument("--stress-target", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.final_f4_release == "collect":
        corpus = write_method_independent_corpus(
            arguments.method_corpus_path, master_seed=METHOD_INDEPENDENT_MASTER_SEED
        )
        print(json.dumps({
            "stage": "method_independent_collection",
            "path": str(arguments.method_corpus_path),
            "completed_stage": corpus["collection_policy"]["completed_stage"],
            "unique_baseline_deadlocks": corpus["unique_baseline_deadlocks"],
            "strata": corpus["strata"],
        }, sort_keys=True))
        return
    if arguments.final_f4_release == "calibrate":
        print(json.dumps(
            calibrate_method_independent_corpus(arguments.method_corpus_path),
            sort_keys=True,
        ))
        return
    if arguments.performance_closure == "profile":
        print(json.dumps(performance_closure_profile(arguments.corpus_path), sort_keys=True))
        return
    if arguments.performance_closure == "differential":
        print(json.dumps(
            performance_closure_differential(arguments.corpus_path), sort_keys=True
        ))
        return
    if arguments.performance_closure == "marginal":
        print(json.dumps(
            performance_closure_marginal(arguments.corpus_path), sort_keys=True
        ))
        return
    if arguments.performance_closure == "n100":
        print(json.dumps(
            formal_n100_performance(
                arguments.seed,
                scope=FORMAL_SCOPE_V1_1,
                time_limit=arguments.time_limit,
            ),
            sort_keys=True,
        ))
        return
    if arguments.formal_scope_v1_1 == "replay":
        analysis = replay_v1_1_deadlock_stress_corpus(arguments.corpus_path)
        print(json.dumps({"stage": "v1_1_stress_replay", **analysis}, sort_keys=True))
        return
    if arguments.formal_scope_v1_1 == "performance":
        print(json.dumps(
            formal_n100_performance(arguments.seed, scope=FORMAL_SCOPE_V1_1),
            sort_keys=True,
        ))
        return
    if arguments.formal_scope_v1_1:
        formal_scope_gate(
            arguments.formal_scope_v1_1,
            tuple(arguments.formal_seeds),
            scope=FORMAL_SCOPE_V1_1,
        )
        return
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
