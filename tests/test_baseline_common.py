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


def test_zr_outcome_taxonomy_uses_evidence_not_termination_string():
    from scripts.run_phase3zr_runtime_audit import outcome
    assert outcome(final_certified=True) == "CERTIFIED_INCUMBENT"
    assert outcome() == "NO_CERTIFIED_INCUMBENT"
    assert outcome(numeric=1) == "NUMERIC_FAILURE"
    assert outcome(final_certified=True, numeric=1) == "NUMERIC_FAILURE"
    assert outcome(exception=True) == "EXECUTION_FAILURE"


def test_zr_observer_same_result_and_candidate_replay():
    from dataclasses import asdict
    from scripts.run_phase3zr_runtime_audit import ReferenceTrace, reconstruct
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
    from scripts.run_phase3zr_runtime_audit import ReferenceTrace
    from mrta_reference.scheduler import FormalReferenceEvaluator
    original = FormalReferenceEvaluator.__call__
    config, _, solution = _solution()
    def fail(row):
        raise RuntimeError("observer sink failed")
    with pytest.raises(RuntimeError, match="observer sink failed"):
        with ReferenceTrace("FIXTURE", sink=fail):
            FormalReferenceEvaluator(FORMAL_SCOPE_V1_1)(solution, config)
    assert FormalReferenceEvaluator.__call__ is original


def test_zr_rejects_forbidden_roles_before_workbook_loading():
    import pytest
    from scripts import run_phase3zr_runtime_audit as zr
    roles = zr.read(zr.historical.ROLES_PATH)
    for role in ("V2_TRAIN_POOL", "V2_VALIDATION", "ID_TEST_SEALED"):
        entry = next(e for e in roles["workbooks"] if e["new_v2_role"] == role)
        def forbidden(*args, **kwargs):
            pytest.fail("forbidden workbook was opened")
        with pytest.raises(PermissionError):
            zr.load_parents(entry, loader=forbidden)
    allowed = next(e for e in roles["workbooks"] if e["new_v2_role"] == zr.ROLE)
    allowed = {**allowed, "sheet_name": "fixture", "instance_id": "fixture"}
    with pytest.raises(RuntimeError, match="allowed loader reached"):
        zr.load_parents(allowed, loader=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("allowed loader reached")))
    with pytest.raises(ValueError, match="Z11"):
        zr.load_parents(allowed, forensic=True, loader=forbidden)


def test_zr_persistence_cannot_write_history_or_arbitrary_paths():
    import pytest
    from scripts import run_phase3zr_runtime_audit as zr
    for path in (zr.historical.ARTIFACT_PATH, zr.historical.PROTOCOL_PATH, zr.ROOT / "README.md"):
        with pytest.raises(ValueError, match="cannot write"):
            zr.write(path, {})


def test_zr_external_forensic_rejects_other_methods_seeds_and_early_access(monkeypatch):
    import pytest
    from scripts import run_phase3zr_runtime_audit as zr
    for method, seed in ((zr.METHODS[0], 20261015), (zr.METHODS[1], 20261021)):
        with pytest.raises(PermissionError, match="Z11/HGA"):
            zr.run_one({}, method, seed, forensic=True)
    monkeypatch.setattr(zr, "read", lambda path: {"status": "RUNNING"})
    with pytest.raises(PermissionError, match="must precede"):
        zr.run_one({}, zr.METHODS[1], 20261015, forensic=True)


def test_zr_percentiles_and_runtime_threshold_boundaries():
    from scripts.run_phase3zr_runtime_audit import distribution, runtime_gate
    assert distribution([])["p95"] is None
    assert distribution([0.0, 100.0])["p95"] == 95.0
    summary = {"runs": 72, "reference_distribution": {"ALL": {"count": 1, "p95": 8.0, "p99": 30.0, "max": 60.0}},
               "run_runtime": {"max": 120.0}, "numeric_failure": 0, "certifier_mismatch": 0, "execution_failure": 0}
    assert runtime_gate(summary)
    summary["reference_distribution"]["ALL"]["max"] = 60.0001
    assert not runtime_gate(summary)


def test_zr_observer_preserves_protected_certifier_identity_in_real_alns_initialization():
    from scripts.run_phase3zr_runtime_audit import ReferenceTrace
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


def test_zr_authorization_accepts_legal_no_certified_outcomes_but_rejects_execution_failure():
    from scripts import run_phase3zr_runtime_audit as zr
    runtime_set = {"instances": [{"instance_id": str(i), "tier": "LARGE" if i < 8 else "MEDIUM"} for i in range(12)]}
    rows = [{"instance_id": e["instance_id"], "method_id": m, "solver_seed": seed, "data_role": zr.ROLE,
             "solver_outcome_class": "NO_CERTIFIED_INCUMBENT"}
            for e in runtime_set["instances"] for m in zr.METHODS for seed in zr.SEEDS]
    summary = {"runs": 72, "reference_distribution": {"ALL": {"count": 1, "p95": 8.0, "p99": 30.0, "max": 60.0}},
               "run_runtime": {"max": 120.0}, "numeric_failure": 0, "certifier_mismatch": 0, "execution_failure": 0}
    audit = {"records": rows, "summary": summary, "status": "COMPLETE", "external_frozen_stress": {"execution_status": "COMPLETED"}}
    slow = {"status": "COMPLETE", "profiles": [{"observational_equivalence": True}], "frozen_set": {"calls": [{}]}}
    h_rows = [{"instance_id": e["instance_id"], "method_id": zr.METHODS[1], "solver_seed": seed, "data_role": zr.ROLE}
              for e in runtime_set["instances"] if e["tier"] == "LARGE" for seed in zr.HGA_SEEDS]
    hga = {"records": h_rows, "summary": dict(summary), "status": "COMPLETE", "classification": "SYSTEMIC_NO_CERTIFIED"}
    decision = zr.assess_decision(audit, slow, hga, runtime_set, regression_pass=True, historical_unchanged=True)
    assert decision["PHASE4_0_AUTHORIZED"] == "YES"
    assert decision["PHASE3Z_HISTORICAL_STATUS"] == "FAIL_179_OF_180_CERTIFIED"
    hga["summary"]["execution_failure"] = 1
    decision = zr.assess_decision(audit, slow, hga, runtime_set, regression_pass=True, historical_unchanged=True)
    assert decision["PHASE4_0_AUTHORIZED"] == "NO"


def test_zr_finalized_result_remains_read_only_after_directory_migration(monkeypatch, capsys):
    import pytest
    from scripts import run_phase3zr_runtime_audit as zr
    audit = {"finalized_at": "completed before migration",
             "final_regression": {"stdout": "original regression output"},
             "decision": {"PHASE3ZR_EXECUTION_STATUS": "PASS"}}
    monkeypatch.setattr(zr, "require_prepared", lambda: None)
    monkeypatch.setattr(zr, "read", lambda path: audit if path == zr.AUDIT else pytest.fail("completed artifact reread"))
    monkeypatch.setattr(zr, "write", lambda *args: pytest.fail("completed artifact rewritten"))
    monkeypatch.setattr(zr.subprocess, "run", lambda *args, **kwargs: pytest.fail("completed experiment rerun"))
    zr.finalize()
    output = capsys.readouterr().out
    assert "original regression output" in output
    assert '"PHASE3ZR_EXECUTION_STATUS": "PASS"' in output


def test_zr_replay_tables_lossless_dedup_and_independent_expansion():
    from copy import deepcopy
    from scripts import run_phase3zr_runtime_audit as zr
    config, _, solution = _solution()
    replay = zr.historical.json_ready(zr.payload(solution, config, ((0,), (), (0,), ())))
    replay["extra_metadata"] = {"preserved": True}
    changed = deepcopy(replay)
    changed["directions"][0][0] = 1
    original = {"records": [{"reference_trace": [
        {"replay": replay, "scheduler_duration": 0.1},
        {"replay": deepcopy(replay), "scheduler_duration": 0.2},
        {"replay": changed, "scheduler_duration": 0.3}]}],
        "external_frozen_stress": {"reference_trace": [{"replay": deepcopy(replay)}]},
        "observer_startup_attempts": [{"reference_trace": [], "error": "retained"}],
        "decision": {"PHASE3ZR_EXECUTION_STATUS": "PASS"}}
    before = deepcopy(original)
    packed = zr.pack_replays(original)
    assert original == before
    assert len(packed["replay_tables"]["parents"]) == 1
    assert len(packed["replay_tables"]["metadata"]) == 1
    assert len(packed["replay_tables"]["replays"]) == 2
    restored = zr.unpack_replays(packed)
    assert restored == before
    assert zr.pack_replays(restored) == packed
    restored["records"][0]["reference_trace"][0]["replay"]["parents"][0]["start"][0] += 1
    assert restored["records"][0]["reference_trace"][1]["replay"] == replay
    assert original == before
    assert zr.unpack_replays(packed) == before
    for row in zr.unpack_replays(packed)["records"][0]["reference_trace"]:
        rebuilt, cfg, dirs = zr.reconstruct(row["replay"])
        assert rebuilt.canonical_hash == solution.canonical_hash
        assert cfg.scientific_hash == config.scientific_hash


def test_zr_replay_tables_reject_invalid_references_and_unknown_formats():
    import pytest
    from scripts import run_phase3zr_runtime_audit as zr
    for index in (-1, 0, True, "0"):
        bad = {"storage_format": zr.STORAGE_FORMAT, "replay_tables": {"replays": []},
               "records": [{"replay_ref": index}]}
        with pytest.raises(ValueError, match="invalid replays reference"):
            zr.unpack_replays(bad)
    with pytest.raises(ValueError, match="unsupported"):
        zr.unpack_replays({"storage_format": "unsupported", "replay_tables": {}})
    with pytest.raises(ValueError, match="reserved replay_ref"):
        zr.pack_replays({"records": [{"replay_ref": 0}]})


def test_zr_writer_compacts_only_audit_and_reads_legacy_json(tmp_path, monkeypatch):
    from scripts import run_phase3zr_runtime_audit as zr
    config, _, solution = _solution()
    original = {"records": [{"reference_trace": [{"replay": zr.historical.json_ready(
        zr.payload(solution, config, ((0,), (), (0,), ())))}]}]}
    path = tmp_path / "audit.json"
    monkeypatch.setattr(zr, "AUDIT", path)
    zr.write(path, original)
    assert "replay_ref" in zr.historical.read_json(path)["records"][0]["reference_trace"][0]
    assert zr.read(path) == original
    zr.write(path, zr.read(path))
    assert zr.read(path) == original
    path.write_text(__import__("json").dumps(original), encoding="utf-8")
    assert zr.read(path) == original
