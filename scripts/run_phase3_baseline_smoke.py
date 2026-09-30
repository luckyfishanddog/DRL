from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns
from mrta_baselines.common import CommonBaselineEvaluator, canonical_config_hash
from mrta_data.phase3_split import ROLE_DEVELOPMENT, assert_solver_access_allowed
from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.model import ScientificConfig
from mrta_reference.provenance import REPOSITORY_ID, compute_source_tree_hash
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi
from mrta_search.initialization import _construct_rail_serial_bootstrap, _patterns


ROOT = Path(__file__).resolve().parents[1]
ALNS_METHOD_ID = "SA_OI_ALNS_INIT_POLICY_V2"
SEEDS = (20260928, 20260929, 20260930)


def _select_instances(
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
        if not candidates:
            raise ValueError(f"no distinct DEVELOPMENT_CONSUMED instance for {tier}")
        entry = min(
            candidates,
            key=lambda item: (
                abs(int(item["actual_weld_count"]) - target),
                item["instance_geometry_hash"],
                item["instance_id"],
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


def _baseline_record(result, *, entry, seed, budget, provenance):
    initial_events = [event for event in result.best_events if "INITIAL" in event.source]
    return {
        **provenance,
        "method_id": result.method_id,
        "method_config_hash": result.method_config_hash,
        "solver_seed": seed,
        "instance_id": entry["instance_id"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "relative_path": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "actual_weld_count": entry["actual_weld_count"],
        "phase3_role": entry["phase3_role"],
        "requested_time_limit_s": budget,
        "initialization_status": (
            "SUCCESS" if initial_events else "NO_CERTIFIED_NATIVE_INITIAL"
        ),
        "time_to_first_certified": result.time_to_first_certified,
        "initial_cmax": initial_events[0].cmax if initial_events else None,
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
        "overshoot": result.overshoot,
        "initialization_time": result.initialization_time,
        "search_time": result.search_time,
        "scheduler_time": result.accounting["scheduler_time"],
        "local_search_time": result.accounting["local_search_time"],
        "factorial_local_search_time": result.accounting["factorial_local_search_time"],
        "direction_dp_calls": result.accounting["direction_dp_calls"],
        "final_status": result.status.value,
        "final_schedule_status": (
            None if result.schedule is None else result.schedule.status.value
        ),
        "final_certified": result.final_certified,
        "best_source_operator": result.best_source,
        "diagnostics": list(result.diagnostics),
    }


def _alns_record(result, *, entry, seed, budget, provenance, method_config_hash):
    records = result.stats.reference_records
    initial = result.initialization.schedule
    return {
        **provenance,
        "method_id": ALNS_METHOD_ID,
        "method_config_hash": method_config_hash,
        "solver_seed": seed,
        "instance_id": entry["instance_id"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "relative_path": entry["relative_path"],
        "sheet_name": entry["sheet_name"],
        "tier": entry["tier"],
        "actual_weld_count": entry["actual_weld_count"],
        "phase3_role": entry["phase3_role"],
        "requested_time_limit_s": budget,
        "initialization_status": result.initialization.status.value,
        "time_to_first_certified": (
            result.stats.best_events[0][0] if result.stats.best_events else None
        ),
        "initial_cmax": None if initial is None else initial.cmax,
        "best_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "cmax_at_5": result.anytime.get(5.0, {}).get("cmax"),
        "cmax_at_30": result.anytime.get(30.0, {}).get("cmax"),
        "cmax_at_60": (
            result.anytime.get(60.0, {}).get("cmax") if budget >= 60.0 else None
        ),
        "iterations": result.stats.iterations,
        "candidate_count": result.stats.constructed,
        "reference_calls": len(records),
        "certifier_calls": sum(item["status"] == "FEASIBLE" for item in records) + 1,
        "baseline_DEADLOCK": sum(bool(item["baseline_deadlock"]) for item in records),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in records
        ),
        "remaining_DEADLOCK": sum(item["status"] == "DEADLOCK" for item in records),
        "runtime": result.runtime,
        "overshoot": result.stats.overshoot,
        "initialization_time": result.stats.init_time,
        "search_time": max(0.0, result.runtime - result.stats.init_time),
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
            None if not result.stats.improvements_by_family else max(
                result.stats.improvements_by_family,
                key=result.stats.improvements_by_family.get,
            )
        ),
        "diagnostics": [],
    }


def _common_certified_seed_diagnostic(parents, *, entry, budget):
    config = ScientificConfig()
    solution = _construct_rail_serial_bootstrap(parents, _patterns(parents, config), config)
    evaluator = CommonBaselineEvaluator(
        parents,
        config,
        FORMAL_SCOPE_V1_1,
        time_limit=budget,
        checkpoints=(),
    )
    candidate = evaluator.evaluate(
        solution, source="COMMON_CERTIFIED_RAIL_SERIAL_SEED"
    )
    return {
        "instance_id": entry["instance_id"],
        "relative_path": entry["relative_path"],
        "seed_source": "RAIL_SERIAL_BOOTSTRAP",
        "seed_certified": bool(
            candidate.certification and candidate.certification.certified
        ),
        "seed_status": candidate.status.value,
        "seed_canonical_hash": solution.canonical_hash,
        "seed_cmax": None if candidate.metrics is None else candidate.metrics.cmax,
        "runtime": evaluator.elapsed,
        "reference_calls": evaluator.accounting.reference_calls,
        "certifier_calls": evaluator.accounting.certifier_calls,
        "injection_policy": "DIAGNOSTIC_ONLY_NATIVE_INITIALIZATIONS_PRESERVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 three-method development smoke")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--include-60", action="store_true")
    parser.add_argument("--source-commit-label", required=True)
    parser.add_argument(
        "--output", default="data/development/phase3_baseline_smoke_v1.json"
    )
    args = parser.parse_args()
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    ppo_root = Path(args.ppo_root).resolve()
    dataset = json.loads(
        (ROOT / "data/manifests/PPO_DATASET_MANIFEST_V1.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (ROOT / "data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json").read_text(encoding="utf-8")
    )
    selected = _select_instances(dataset, split)
    source_tree_hash = compute_source_tree_hash(ROOT)
    base_provenance = {
        "repository_id": REPOSITORY_ID,
        "source_commit": args.source_commit_label,
        "source_tree_hash": source_tree_hash,
        "scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "dataset_manifest_hash": dataset["dataset_manifest_hash"],
        "phase3_split_hash": split["phase3_split_hash"],
        "development_only": True,
        "commit_verified": False,
    }
    budgets = [args.budget] + ([60.0] if args.include_60 and args.budget != 60.0 else [])
    records = []
    common_seed_diagnostics = []
    for budget in budgets:
        for entry in selected:
            instance = load_ppo_platform_instance(
                ppo_root / entry["relative_path"],
                entry["sheet_name"],
                ppo_root=ppo_root,
                instance_id=entry["instance_id"],
            )
            parents = to_parent_welds(instance)
            common_seed_diagnostics.append(
                _common_certified_seed_diagnostic(
                    parents, entry=entry, budget=budget
                )
            )
            for seed in SEEDS:
                alns_config = SearchConfig(
                    time_limit=budget, checkpoints=(5.0, 30.0, 60.0)
                )
                alns = run_bounded_sa_oi(
                    parents,
                    ScientificConfig(),
                    alns_config,
                    seed=seed,
                    scope=FORMAL_SCOPE_V1_1,
                    source_commit=args.source_commit_label,
                    allow_unverified_source=True,
                    formal_result=False,
                )
                records.append(
                    _alns_record(
                        alns,
                        entry=entry,
                        seed=seed,
                        budget=budget,
                        provenance=base_provenance,
                        method_config_hash=canonical_config_hash(alns_config),
                    )
                )
                hga = run_adapted_hga(
                    parents,
                    ScientificConfig(),
                    AdaptedHGAConfig(),
                    seed=seed,
                    time_limit=budget,
                    scope=FORMAL_SCOPE_V1_1,
                )
                records.append(
                    _baseline_record(
                        hga,
                        entry=entry,
                        seed=seed,
                        budget=budget,
                        provenance=base_provenance,
                    )
                )
                wag = run_adapted_wag_vns(
                    parents,
                    ScientificConfig(),
                    AdaptedWAGConfig(),
                    seed=seed,
                    time_limit=budget,
                    scope=FORMAL_SCOPE_V1_1,
                )
                records.append(
                    _baseline_record(
                        wag,
                        entry=entry,
                        seed=seed,
                        budget=budget,
                        provenance=base_provenance,
                    )
                )
    payload = {
        "smoke_id": "PHASE3_BASELINE_SMOKE_V1",
        "selection_policy": (
            "DEVELOPMENT_CONSUMED only; distinct workbooks; N20-30/N50-60/N80-90; "
            "closest to 25/55/85 then geometry hash"
        ),
        "solver_seeds": list(SEEDS),
        "selected_instances": [
            {
                key: entry[key]
                for key in (
                    "tier",
                    "instance_id",
                    "relative_path",
                    "sheet_name",
                    "actual_weld_count",
                    "instance_geometry_hash",
                    "phase3_role",
                )
            }
            for entry in selected
        ],
        "provenance": base_provenance,
        "common_certified_seed_diagnostics": common_seed_diagnostics,
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))
    return 0 if all(record["final_certified"] for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
