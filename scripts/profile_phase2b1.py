from __future__ import annotations

import argparse
import json

from mrta_reference.model import ParentWeld, ScientificConfig
from mrta_search import SearchConfig, run_bounded_sa_oi


def synthetic_parents(count: int) -> tuple[ParentWeld, ...]:
    """Deterministic development-only geometry; never used as formal data."""
    parents = []
    per_rail = (count + 1) // 2
    left_count = (per_rail + 1) // 2
    right_count = per_rail // 2
    for index in range(count):
        rail_index = index // 2
        if rail_index % 2 == 0:
            local = rail_index // 2
            x = 0.5 + 8.5 * local / max(1, left_count - 1)
        else:
            local = rail_index // 2
            x = 19.3 - 8.5 * local / max(1, right_count - 1)
        y0, y1 = ((9.0, 10.0) if index % 2 == 0 else (2.0, 3.0))
        parents.append(
            ParentWeld(f"p{index:03d}", (x, y0), (x, y1))
        )
    return tuple(parents)


def profile(count: int, seconds: float, seed: int) -> dict[str, object]:
    result = run_bounded_sa_oi(
        synthetic_parents(count),
        ScientificConfig(),
        SearchConfig(
            m=64,
            kdp=8,
            kref=2,
            max_iterations=100_000,
            time_limit=seconds,
        ),
        seed=seed,
    )
    stats = result.stats
    initial = (
        None
        if result.initialization.schedule is None
        else result.initialization.schedule.cmax
    )
    best = None if result.best_schedule is None else result.best_schedule.cmax
    attempted = sum(stats.attempted_by_move.values())
    valid = sum(stats.valid_by_move.values())
    accepted = sum(stats.accepted_by_move.values())
    return {
        "N": count,
        "budget_seconds": seconds,
        "seed": seed,
        "status": result.status.value,
        "runtime_seconds": result.runtime,
        "initial_cmax": initial,
        "best_cmax": best,
        "improvement": None if initial is None or best is None else initial - best,
        "iterations": stats.iterations,
        "candidate_per_second": stats.raw_attempts / result.runtime if result.runtime else None,
        "Nref": stats.nref,
        "Nref_per_second": stats.nref / result.runtime if result.runtime else None,
        "scheduler_mean": stats.scheduler_mean,
        "scheduler_p95": stats.scheduler_p95,
        "deadlock_rate": stats.n_deadlock / stats.nref if stats.nref else None,
        "initialization_failed": result.status.value == "INITIALIZATION_FAILED",
        "move_valid_rate": valid / attempted if attempted else None,
        "move_acceptance_rate": accepted / valid if valid else None,
        "certified": bool(result.final_certification and result.final_certification.certified),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2B-1 development profiling (not a formal benchmark)"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=(20, 50, 100))
    parser.add_argument("--budgets", nargs="+", type=float, default=(0.2, 1.0, 5.0))
    parser.add_argument("--seed", type=int, default=20260928)
    arguments = parser.parse_args()
    rows = [
        profile(size, budget, arguments.seed)
        for size in arguments.sizes
        for budget in arguments.budgets
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
