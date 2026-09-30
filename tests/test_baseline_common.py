from __future__ import annotations

from mrta_baselines.common import (
    BaselineBestEvent,
    BaselineStatus,
    CommonBaselineEvaluator,
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

