from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import runpy
import subprocess

from mrta_exact import ExactSolveStatus, exact_schedule_from_templates, solve_exact_micro
from mrta_reference.model import Operation, OperationKind, ParentWeld, ScientificConfig
from mrta_reference.scheduler import (
    SchedulerProfile,
    build_operation_templates,
    build_robot_routes,
    reference_schedule_from_templates_optimized,
    reference_schedule_from_templates_slow,
    reference_schedule_optimized,
    reference_schedule_slow,
)
from mrta_search import SearchConfig, micro_gap_decomposition, run_bounded_sa_oi
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.initialization import InitializationStrategy, _construct, _patterns

from profile_phase2b1 import synthetic_parents


def formal_scope_gate(stage: str, seeds: tuple[int, ...]) -> None:
    """Development-only freeze evidence; JSON lines on stdout, no dataset files."""
    from mrta_reference.scope import FORMAL_SCOPE_V1, RunScientificIdentity
    from mrta_reference.scheduler import (
        reference_schedule_from_templates_formal, _bounded_dispatch_recovery,
    )
    from mrta_reference.certifier import certify_template_schedule
    scope = FORMAL_SCOPE_V1
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    def emit(row):
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    emit({"stage": "identity", **asdict(RunScientificIdentity.from_scope(scope, ScientificConfig(), commit)),
          "development_only": True, "source_worktree_dirty": bool(subprocess.check_output(
              ["git", "status", "--porcelain"], text=True).strip())})
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
                                      seed=seed, scope=scope, source_commit=commit)
            if not result.final_certification or not result.final_certification.certified:
                raise RuntimeError(f"uncertified quality case {name}")
            emit({"stage": "quality", "case": name, "seed": seed,
                  "identity": asdict(result.stats.scientific_identity),
                  "development_C_star": exact.best_cmax, "development_exact_status": exact.status.value,
                  "formal_initial_Cref": result.initialization.schedule.cmax,
                  "formal_Cref": result.best_schedule.cmax, "formal_exact_gap": None,
                  "status": result.status.value, "certified": result.final_certification.certified,
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
                                              scope=scope, source_commit=commit)
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
                          "deadlock": s.n_deadlock, "repair_time": s.repair_time,
                          "reference_time": s.reference_scheduler_time, "certifier_time": s.certifier_time})


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
    arguments = parser.parse_args()
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
