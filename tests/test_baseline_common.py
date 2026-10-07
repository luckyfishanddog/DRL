from __future__ import annotations

from mrta_baselines.common import (
    BaselineBestEvent,
    BaselineStatus,
    CommonBaselineEvaluator,
    solution_telemetry,
)
from mrta_reference.model import ParentWeld, Route, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_reference.solution import canonicalize


def _solution():
    config = ScientificConfig()
    parents = (
        ParentWeld("upper", (2.0, 9.0), (3.0, 9.0)),
        ParentWeld("lower", (14.0, 3.0), (15.0, 3.0)),
    )
    patterns = tuple(
        SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents
    )
    solution = canonicalize(
        parents,
        patterns,
        (
            Route(0, ("upper::whole",)),
            Route(1, ()),
            Route(2, ("lower::whole",)),
            Route(3, ()),
        ),
        config,
    )
    return config, parents, solution


def test_common_baseline_path_uses_formal_scope_direction_and_certifier():
    config, parents, solution = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=2.0, checkpoints=(0.0, 2.0)
    )
    candidate = evaluator.evaluate(solution, source="FIXTURE")
    assert candidate.status is BaselineStatus.COMPLETED
    assert candidate.schedule is not None
    assert candidate.schedule.scope_id == FORMAL_SCOPE_V1_1.scope_id
    assert candidate.schedule.scope_hash == FORMAL_SCOPE_V1_1.scope_hash
    assert candidate.certification is not None and candidate.certification.certified
    assert evaluator.accounting.reference_calls == 1
    assert evaluator.accounting.certifier_calls == 1


def test_checkpoint_semantics_do_not_backfill_late_feasible_solution():
    config, parents, _ = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=30.0, checkpoints=(5.0, 30.0, 60.0)
    )
    evaluator.best_events.extend(
        (BaselineBestEvent(6.0, 100.0, "LATE"), BaselineBestEvent(20.0, 90.0, "BETTER"))
    )
    assert evaluator.checkpoint_values() == {5.0: None, 30.0: 90.0, 60.0: None}


def test_failure_has_no_penalty_cmax_and_is_separately_classified():
    config, parents, _ = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=1.0
    )
    evaluator.reject_construction("fixture rejection")
    result = evaluator.finish(
        method_id="FIXTURE",
        method_config_hash="fixture",
        iterations=0,
        initialization_time=0.0,
    )
    assert result.status is BaselineStatus.INITIALIZATION_FAILED
    assert result.metrics is None
    assert result.solution is None
    assert result.termination_reason == "COMPLETED_OTHER"


def test_initial_solution_telemetry_hash_is_canonical_deterministic_and_source_free():
    config, _, solution = _solution()
    directions = ((0,), (), (0,), ())
    first = solution_telemetry(solution, directions, config)
    second = solution_telemetry(solution, directions, config)
    assert first == second
    assert first["initial_solution_hash"] == solution.canonical_hash
    assert first["initial_robot_block_counts"] == [1, 0, 1, 0]
    assert len(first["initial_route_hashes"]) == 4


def test_zr_observer_same_result_and_candidate_replay():
    from dataclasses import asdict
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config, parents, solution = _solution()
    expected = FormalReferenceEvaluator(FORMAL_SCOPE_V2)(solution, config)
    for level in (0, 1, 2):
        with ReferenceTrace("FIXTURE", level) as trace:
            evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V2, time_limit=2.0)
            candidate = evaluator.evaluate(solution, source="FIXTURE_SOURCE")
        assert asdict(candidate.schedule) == asdict(expected)
        if level:
            row = trace.serialize({"instance_id": "fixture", "instance_geometry_hash": "fixture", "tier": "SMALL"})[0]
            assert row["scheduler_duration"] > 0
            assert row["recovery_time"] == 0
            assert row["certified"] is True
            assert row["candidate_source"] == "FIXTURE_SOURCE"
            assert abs(sum(row[k] for k in ("preparation_time", "baseline_dispatch_time", "recovery_time", "packaging_time"))-row["scheduler_duration"]) < 1e-8
            recovered, cfg, directions = reconstruct(row["replay"])
            assert recovered.canonical_hash == solution.canonical_hash
            assert cfg.scientific_hash == config.scientific_hash
            assert tuple(directions.values()) == expected.directions
        else:
            assert trace.rows == []

def test_zr_observer_exception_explicitly_fails_closed_and_restores():
    import pytest
    from mrta_reference.scheduler import FormalReferenceEvaluator
    original = FormalReferenceEvaluator.__call__
    config, _, solution = _solution()
    def fail(row):
        raise RuntimeError("observer sink failed")
    with pytest.raises(RuntimeError, match="observer sink failed"):
        with ReferenceTrace("FIXTURE", sink=fail):
            FormalReferenceEvaluator(FORMAL_SCOPE_V1_1)(solution, config)
    assert FormalReferenceEvaluator.__call__ is original

def test_zr_observer_preserves_protected_certifier_identity_in_real_alns_initialization():
    from mrta_search.initialization import build_initial_solution, InitializationStatus
    from mrta_search.stats import SearchStats
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.certifier import certify_schedule
    import mrta_search.initialization as initialization
    config, parents, _ = _solution()
    with ReferenceTrace("SA_OI_ALNS_V2") as trace:
        result = build_initial_solution(parents, config, SearchStats(FORMAL_SCOPE_V2.scope_id, 0),
                    scope=FORMAL_SCOPE_V2, construction_budget=2, kinit_ref=2)
        assert initialization.certify_schedule is certify_schedule
    assert result.status is InitializationStatus.SUCCESS
    assert trace.rows and any(row["certified"] for row in trace.rows)

import time,traceback,json,hashlib
from functools import wraps
from dataclasses import asdict
from mrta_reference import scheduler
from mrta_reference.model import ParentWeld,Route,SplitPattern,SplitKind,Rail,ScientificConfig,ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize
from mrta_search.stats import SearchStats
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
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
            rows.append(json.loads(json.dumps(row)))
        return rows
