from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mrta_data.ppo_instances import load_ppo_platform_instance, to_parent_welds
from mrta_reference.model import ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_search.pipeline import SearchConfig, run_bounded_sa_oi


def _checkpoint(result, deadline: float, budget: float) -> dict[str, object]:
    if deadline > budget:
        return {"cmax": None, "reason": "CHECKPOINT_BEYOND_REQUESTED_BUDGET"}
    return result.anytime[deadline]


def _run(entry: dict[str, Any], root: Path, budget: float, source_commit: str) -> dict[str, Any]:
    instance = load_ppo_platform_instance(
        root / entry["relative_path"],
        entry["sheet_name"],
        ppo_root=root,
        instance_id=entry["instance_id"],
    )
    parents = to_parent_welds(instance)
    result = run_bounded_sa_oi(
        parents,
        ScientificConfig(),
        SearchConfig(time_limit=budget),
        seed=20260929,
        scope=FORMAL_SCOPE_V1_1,
        source_commit=source_commit,
        allow_unverified_source=True,
        formal_result=False,
    )
    records = result.stats.reference_records
    initial_schedule = result.initialization.schedule
    final_certified = bool(
        result.final_certification and result.final_certification.certified
    )
    return {
        "tier": entry["tier"],
        "instance_id": entry["instance_id"],
        "actual_weld_count": entry["actual_weld_count"],
        "total_weld_length_m": entry["total_weld_length_m"],
        "dataset_manifest_hash": entry["dataset_manifest_hash"],
        "instance_geometry_hash": entry["instance_geometry_hash"],
        "solver_seed": 20260929,
        "requested_time_limit_s": budget,
        "initialization_status": result.initialization.status.value,
        "initial_cmax": None if initial_schedule is None else initial_schedule.cmax,
        "best_cmax": None if result.best_metrics is None else result.best_metrics.cmax,
        "cmax_at_1": _checkpoint(result, 1.0, budget),
        "cmax_at_5": _checkpoint(result, 5.0, budget),
        "cmax_at_30": _checkpoint(result, 30.0, budget),
        "iterations": result.stats.iterations,
        "Nref": result.stats.nref,
        "init_reference_calls": result.stats.init_reference_calls,
        "total_reference_calls": len(records),
        "baseline_DEADLOCK": sum(bool(item["baseline_deadlock"]) for item in records),
        "recovered": sum(
            bool(item["baseline_deadlock"]) and item["status"] == "FEASIBLE"
            for item in records
        ),
        "remaining_DEADLOCK": sum(item["status"] == "DEADLOCK" for item in records),
        "actual_runtime_s": result.runtime,
        "overshoot_s": result.stats.overshoot,
        "scheduler_time_s": result.stats.reference_scheduler_time,
        "repair_time_s": result.stats.repair_time,
        "certifier_time_s": result.stats.certifier_time,
        "final_schedule_status": (
            None if result.best_schedule is None else result.best_schedule.status.value
        ),
        "final_certification": final_certified,
        "certification_errors": (
            []
            if result.final_certification is None
            else list(result.final_certification.errors)
        ),
        "search_status": result.status.value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 compatibility smoke on frozen PPO instances")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument(
        "--manifest", default="data/manifests/PPO_DATASET_MANIFEST_V1.json"
    )
    parser.add_argument(
        "--smokeset", default="data/manifests/PPO_PHASE3_SMOKESET_V1.json"
    )
    parser.add_argument("--include-30", action="store_true")
    parser.add_argument(
        "--source-commit", default="e4de209d872d46687af4974d030b32904192b906"
    )
    args = parser.parse_args()
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    root = Path(args.ppo_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    smokeset = json.loads(Path(args.smokeset).read_text(encoding="utf-8"))
    by_id = {entry["instance_id"]: entry for entry in manifest["instances"]}
    selected = []
    for selected_entry in smokeset["instances"]:
        entry = dict(by_id[selected_entry["instance_id"]])
        entry["tier"] = selected_entry["tier"]
        entry["dataset_manifest_hash"] = manifest["dataset_manifest_hash"]
        selected.append(entry)
    results = [_run(entry, root, 5.0, args.source_commit) for entry in selected]
    if args.include_30:
        results.extend(
            _run(entry, root, 30.0, args.source_commit)
            for entry in selected
            if entry["tier"] in {"medium", "large"}
        )
    payload = {
        "smokeset_id": smokeset["smokeset_id"],
        "dataset_manifest_id": manifest["dataset_manifest_id"],
        "dataset_manifest_hash": manifest["dataset_manifest_hash"],
        "formal_scope_id": FORMAL_SCOPE_V1_1.scope_id,
        "formal_scope_hash": FORMAL_SCOPE_V1_1.scope_hash,
        "deadlock_rollout_budget": FORMAL_SCOPE_V1_1.deadlock_rollout_budget,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["final_certification"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
