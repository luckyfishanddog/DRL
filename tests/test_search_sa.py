from __future__ import annotations

import random

import pytest
from pathlib import Path
from mrta_reference.provenance import SourceProvenance, REPOSITORY_ID, compute_source_tree_hash


def _development_provenance():
    return SourceProvenance(REPOSITORY_ID, "YR_TEST_FIXTURE",
                            compute_source_tree_hash(Path(__file__).resolve().parents[1]), True, False)

from mrta_exact import exact_schedule_from_templates, solve_exact_micro
from mrta_reference.model import CandidateKey, CandidateMove, MoveType, OperationKind, ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import build_operation_templates, build_robot_routes, reference_schedule
from mrta_reference.solution import canonicalize
from mrta_search.direction import ConstrainedDirectionResult, DirectionStatus, optimize_directions_with_initial_feasibility
from mrta_search.neighborhood import ScreenedCandidate
from mrta_search.pipeline import DirectionEvaluatedCandidate, SearchConfig, SearchStatus, evaluate_iteration, micro_gap_decomposition, rerank_c3, run_bounded_sa_oi, sa_accept
from mrta_search.stats import SearchStats
from mrta_search.pipeline import phase3_alns_config


# Historical Q1-Q5 quality fixtures retain their original 0.2 feasible domain.
FAST = ScientificConfig(delta_x=.20, delta_y=.20, weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _base():
    parents = tuple(
        ParentWeld(name, (x, 2.0), (x + 1.0, 2.0))
        for name, x in (("a", 1.0), ("b", 4.0), ("c", 10.0), ("d", 14.0))
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {2: ("a::whole", "b::whole"), 3: ("c::whole", "d::whole")},
        FAST,
    )


def _family_candidates():
    from dataclasses import replace
    base = _base()
    rows = []
    # The global cheap order deliberately starves the three pattern families.
    for rank in range(12):
        key = CandidateKey(0, MoveType.INTRA_RELOCATE, ("a",), 2, 2, (0,), (rank,))
        rows.append(ScreenedCandidate(CandidateMove(key), base, (rank,), float(rank), 0.0, 0))
    for ordinal, kind in enumerate((SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT)):
        move_type = MoveType.SPLIT_DEACTIVATE if kind is SplitKind.WHOLE else MoveType.SPLIT_ACTIVATE
        key = CandidateKey(0, move_type, ("a",), 2, 3, (), (ordinal,))
        from mrta_reference.model import Rail
        pattern = None if kind is SplitKind.WHOLE else SplitPattern(
            "a", kind, t=0.5, point_id="MIDPOINT",
            rail=Rail.LOWER if kind is SplitKind.X_SPLIT else None,
        )
        rows.append(ScreenedCandidate(CandidateMove(key, split_pattern=pattern), base,
                                     (100 + ordinal,), float(100 + ordinal), 0.0, 0))
    return rows


def test_yr_classifies_explicit_transition_and_structural_lns():
    from dataclasses import replace
    from mrta_search.lns import CandidateSourceKind, atomic_complete_candidate
    from mrta_search.pipeline import decision_family
    rows = _family_candidates()
    assert [decision_family(row) for row in rows[-3:]] == ["TARGET_WHOLE", "TARGET_Y", "TARGET_X"]
    complete = atomic_complete_candidate(_base(), ((), (), (0, 0), (0, 0)), rows[0], FAST)
    assert decision_family(complete) == "STRUCTURAL"
    lns = replace(complete, source_kind=CandidateSourceKind.LNS_REPAIRED, atomic_move=None)
    assert decision_family(lns) == "STRUCTURAL"
    # An incumbent containing X does not turn a structural move into TARGET_X.
    x_incumbent = replace(rows[0].solution, patterns=(rows[-1].candidate.split_pattern,))
    assert decision_family(replace(rows[0], solution=x_incumbent)) == "STRUCTURAL"


def test_yr_c2_one_per_family_and_remaining_global_fill_with_original_scores():
    from mrta_search.pipeline import decision_family, select_c2_by_family
    rows = _family_candidates()
    selected = select_c2_by_family(list(reversed(rows)), 8, seed=20261005, iteration=0)
    assert len(selected) == SearchConfig().kdp == 8
    assert [decision_family(row) for row in selected[:4]] == ["STRUCTURAL", "TARGET_WHOLE", "TARGET_Y", "TARGET_X"]
    assert selected[4:] == tuple(rows[1:5])
    assert len({id(row) for row in selected}) == 8
    assert select_c2_by_family(rows[:12], 8, seed=1, iteration=0) == tuple(rows[:8])
    assert [row.cheap_score for row in rows] == [(i,) for i in range(12)] + [(100,), (101,), (102,)]


@pytest.mark.parametrize("seed", (20261005, 20261006, 20261007))
@pytest.mark.parametrize("iteration", range(4))
def test_yr_c4_global_exploit_and_deterministic_family_explore(seed, iteration):
    from mrta_search.pipeline import DECISION_FAMILIES, decision_family, select_c4_by_family
    rows = _family_candidates()
    direction = ConstrainedDirectionResult(DirectionStatus.FEASIBLE, ((), (), (0, 0), (0, 0)), 1.0)
    directed = [DirectionEvaluatedCandidate(row, rank, direction) for rank, row in enumerate(rows)]
    selected = select_c4_by_family(directed, 2, seed=seed, iteration=iteration)
    assert len(selected) == SearchConfig().kref == 2
    assert selected[0] is directed[0]
    assert decision_family(selected[1].screened) == DECISION_FAMILIES[(seed + iteration) % 4]
    assert selected[0] is not selected[1]
    assert select_c4_by_family(list(reversed(directed)), 2, seed=seed, iteration=iteration) == selected


def test_yr_c4_empty_family_and_unique_exploit_fallback():
    from mrta_search.pipeline import select_c4_by_family
    rows = _family_candidates()
    direction = ConstrainedDirectionResult(DirectionStatus.FEASIBLE, ((), (), (0, 0), (0, 0)), 1.0)
    directed = [DirectionEvaluatedCandidate(row, rank, direction) for rank, row in enumerate(rows[:3])]
    assert select_c4_by_family(directed, 2, seed=3, iteration=0) == tuple(directed[:2])
    assert select_c4_by_family(directed[:1], 2, seed=0, iteration=0) == tuple(directed[:1])


def test_yr_v1_default_remains_global_shortlist_and_identical_trajectory():
    from mrta_reference.scope import FORMAL_SCOPE_V1_1
    config = SearchConfig(m=32, kdp=8, kref=2, max_iterations=3)
    kwargs = dict(seed=7, scope=FORMAL_SCOPE_V1_1, source_provenance=_development_provenance())
    default = run_bounded_sa_oi(_base().parents, FAST, config, **kwargs)
    legacy = run_bounded_sa_oi(_base().parents, FAST, config, family_access_policy=False, **kwargs)
    assert default.best_solution.canonical_hash == legacy.best_solution.canonical_hash
    assert default.best_schedule.canonical_json() == legacy.best_schedule.canonical_json()
    assert default.stats.proposal_trajectory == legacy.stats.proposal_trajectory
    assert default.stats.reference_status_sequence == legacy.stats.reference_status_sequence


def test_yr_v2_funnel_monotone_caps_and_first_access_times():
    from mrta_reference.scope import FORMAL_SCOPE_V2
    parents = (ParentWeld("access", (1.0, 2.0), (5.0, 2.0)),)
    result = run_bounded_sa_oi(parents, FAST, SearchConfig(max_iterations=4),
                              seed=20261005, scope=FORMAL_SCOPE_V2,
                              enable_x_split=True, source_provenance=_development_provenance())
    stats = result.stats
    assert max(stats.per_iteration_kdp) <= 8
    assert max(stats.per_iteration_nref) <= 4
    for counts in stats.decision_family_funnel.values():
        assert counts["C2_selected"] >= counts["direction_feasible"] >= counts["C4_selected"]
        assert counts["C4_selected"] >= counts["reference_evaluated"] >= counts["certified"]
        assert counts["certified"] >= counts["accepted"] >= counts["global_best_update"]
    assert stats.decision_family_funnel["TARGET_X"]["C2_selected"] > 0
    assert stats.first_x_proposal_time <= stats.first_x_c2_time


def test_phase3_comparison_config_overrides_legacy_iteration_cap():
    assert SearchConfig().max_iterations == 100
    comparison = phase3_alns_config(30.0)
    assert comparison.time_limit == 30.0
    assert comparison.max_iterations >= 100000
    assert comparison.checkpoints == (5.0, 30.0, 60.0)


def test_explicit_iteration_safety_cap_is_reported():
    result = run_bounded_sa_oi(
        _base().parents,
        FAST,
        SearchConfig(max_iterations=0, time_limit=30.0),
        seed=20260929,
    )
    assert result.status is SearchStatus.COMPLETED
    assert result.termination_reason == "ITERATION_LIMIT"


def test_c3_exact_direction_rerank_changes_kref_shortlist() -> None:
    solution = _base()
    rows = []
    for rank, exact_empty in enumerate((9.0, 8.0, 1.0)):
        key = CandidateKey(0, MoveType.INTRA_RELOCATE, ("a",), 2, 2, (0,), (rank,))
        screened = ScreenedCandidate(
            CandidateMove(key),
            solution,
            (10.0, float(rank), 0, (rank,)),
            10.0,
            float(rank),
            0,
        )
        direction = ConstrainedDirectionResult(
            DirectionStatus.FEASIBLE,
            ((0,), (), (), ()),
            exact_empty,
        )
        rows.append(DirectionEvaluatedCandidate(screened, rank, direction))
    shortlist = rerank_c3(rows, 2)
    assert [item.cheap_rank for item in shortlist] == [2, 1]


def test_pipeline_hard_kdp_kref_explicit_directions_and_four_statuses() -> None:
    solution = _base()
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    search_config = SearchConfig(m=32, kdp=4, kref=1, max_iterations=1)
    for status in ScheduleStatus:
        stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 3)
        supplied = []

        def evaluator(candidate, config, *, orientations):
            supplied.append(tuple(tuple(orientations[robot]) for robot in range(4)))
            if status is ScheduleStatus.FEASIBLE:
                return reference_schedule(candidate, config, orientations=orientations)
            return ScheduleResult(status, diagnostics=(status.value,))

        result = evaluate_iteration(
            solution,
            directions,
            FAST,
            search_config,
            stats,
            seed=3,
            reference_evaluator=evaluator,
        )
        assert len(result.c3_candidates) <= 4
        assert len(supplied) <= 1
        assert stats.per_iteration_kdp[-1] <= 4
        assert stats.per_iteration_nref[-1] <= 1
        if supplied:
            assert result.c4_candidates[0].schedule.directions == supplied[0] or status is not ScheduleStatus.FEASIBLE
            assert (
                stats.n_feasible,
                stats.n_deadlock,
                stats.n_infeasible,
                stats.n_numeric_failure,
            )[list(ScheduleStatus).index(status)] == 1
        stats.assert_invariants(kdp=4, kref=1)


def test_sa_acceptance_improve_equal_worse_seeded_and_zero_temperature() -> None:
    assert sa_accept(10.0, 9.0, temperature_value=1.0, cscale=1.0, rng=random.Random(1))[0]
    assert sa_accept(10.0, 10.0 + 1e-10, temperature_value=1.0, cscale=1.0, rng=random.Random(1))[0]
    first = sa_accept(10.0, 11.0, temperature_value=1.0, cscale=10.0, rng=random.Random(42))
    second = sa_accept(10.0, 11.0, temperature_value=1.0, cscale=10.0, rng=random.Random(42))
    assert first == second
    assert 0.0 < first[1] < 1.0
    assert not sa_accept(10.0, 11.0, temperature_value=0.0, cscale=10.0, rng=random.Random(1))[0]


def test_optional_y_activation_enters_search_and_improves_whole_only() -> None:
    parents = (ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),)
    result = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=10),
        seed=0,
    )
    assert result.status is SearchStatus.COMPLETED
    assert result.initialization.schedule.cmax == pytest.approx(6.0)
    assert result.best_schedule.cmax == pytest.approx(4.0)
    assert result.best_solution.patterns[0].kind is SplitKind.Y_SPLIT
    assert result.stats.accepted_by_move[MoveType.SPLIT_ACTIVATE.value] >= 1
    assert result.final_certification.certified


def test_time_limited_run_completes_one_real_iteration_after_initialization() -> None:
    parents = (ParentWeld("one", (1.0, 2.0), (2.0, 2.0)),)
    result = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(max_iterations=100, time_limit=0.0),
        seed=9,
    )
    assert result.status is SearchStatus.COMPLETED
    assert result.stats.iterations == 1
    assert result.stats.actual_runtime is not None
    assert result.stats.overshoot == pytest.approx(result.stats.actual_runtime)
    assert result.final_certification.certified


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_micro_exact_reference_gap_decomposition_and_certified_best(count: int) -> None:
    specs = (
        ("u0", (1.0, 10.0)),
        ("u1", (10.0, 10.0)),
        ("l0", (1.0, 2.0)),
        ("l1", (10.0, 2.0)),
    )[:count]
    parents = tuple(ParentWeld(name, start, (start[0] + 1.0, start[1])) for name, start in specs)
    exact = solve_exact_micro(parents, FAST)
    search = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=28, kdp=4, kref=1, max_iterations=2),
        seed=5,
    )
    gap = micro_gap_decomposition(search, exact, FAST)
    assert gap.total_gap == pytest.approx(gap.search_gap + gap.scheduler_gap)
    assert gap.search_gap >= -1e-9
    assert gap.c_coord <= gap.c_ref + 1e-9
    assert search.final_certification.certified


def test_deadlock_and_numeric_failure_are_never_sa_proposals() -> None:
    solution = _base()
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    for status in (ScheduleStatus.DEADLOCK, ScheduleStatus.NUMERIC_FAILURE, ScheduleStatus.INFEASIBLE):
        stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 9)

        def evaluator(*args, **kwargs):
            return ScheduleResult(status)

        iteration = evaluate_iteration(
            solution,
            directions,
            FAST,
            SearchConfig(m=28, kdp=4, kref=1, max_iterations=1),
            stats,
            seed=9,
            reference_evaluator=evaluator,
        )
        assert iteration.proposal is None


def _nontrivial_quality_cases():
    return {
        "Q1_assignment": (
            (
                ParentWeld("p0", (11.14, 10.0), (12.54, 10.0)),
                ParentWeld("p1", (1.73, 10.0), (3.44, 10.0)),
                ParentWeld("p2", (3.70, 10.0), (4.88, 10.0)),
            ),
            7,
            1,
            MoveType.INTER_RELOCATE,
        ),
        "Q2_route_order": (
            (
                ParentWeld("p0", (7.133, 8.048), (10.143, 11.5)),
                ParentWeld("p1", (10.03, 8.264), (10.365, 8.484)),
                ParentWeld("p2", (13.385, 9.169), (12.862, 9.463)),
                ParentWeld("p3", (12.928, 8.209), (13.724, 8.133)),
            ),
            43,
            8,
            MoveType.TWO_OPT,
        ),
        "Q3_direction": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
            ),
            3,
            20,
            None,
        ),
        "Q4_optional_y": (
            (ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),),
            0,
            10,
            MoveType.SPLIT_ACTIVATE,
        ),
        "Q5_interference_wait": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
                ParentWeld("d", (2.31, 10.0), (2.59, 10.0)),
            ),
            37,
            12,
            MoveType.INTER_RELOCATE,
        ),
    }


def _fixed_coordination_cmax(solution, directions) -> float:
    routes = build_robot_routes(
        solution, FAST, {robot: directions[robot] for robot in range(4)}
    )
    exact = exact_schedule_from_templates(
        {
            route.robot_id: build_operation_templates(route, FAST)
            for route in routes
        },
        FAST,
        directions=directions,
    )
    assert exact.schedule is not None
    assert exact.schedule.cmax is not None
    return exact.schedule.cmax


@pytest.mark.parametrize(
    "case_name", tuple(_nontrivial_quality_cases())
)
def test_q1_q5_nontrivial_initial_gap_and_backbone_quality(case_name: str) -> None:
    parents, seed, iterations, expected_move = _nontrivial_quality_cases()[case_name]
    exact = solve_exact_micro(parents, FAST)
    search = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=iterations),
        seed=seed,
    )
    assert search.initialization.solution is not None
    assert search.initialization.directions is not None
    assert search.initialization.schedule is not None
    assert search.initialization.schedule.cmax is not None
    initial_coord = _fixed_coordination_cmax(
        search.initialization.solution, search.initialization.directions
    )
    assert exact.best_cmax is not None
    initial_gap = initial_coord - exact.best_cmax
    assert initial_gap > 1.0e-6
    final_gap = micro_gap_decomposition(search, exact, FAST)

    if expected_move is None:
        assert case_name == "Q3_direction"
        assert final_gap.search_gap < initial_gap - 1.0e-6
        assert search.stats.direction_improvements >= 1
        assert search.stats.improvements_by_family["DIRECTION_REFINEMENT"] >= 1
    else:
        assert final_gap.search_gap < initial_gap - 1.0e-6
        assert search.stats.best_improvement_by_move[expected_move.value] >= 1

    if case_name == "Q2_route_order":
        assert [set(route.block_ids) for route in search.initialization.solution.routes] == [
            set(route.block_ids) for route in search.best_solution.routes
        ]
    if case_name == "Q5_interference_wait":
        assert any(
            operation.kind is OperationKind.WAIT
            for operation in search.initialization.schedule.operations
        )
    assert search.final_certification is not None
    assert search.final_certification.certified

def test_ranker_features_are_pure_stage_available_and_geometrically_correct(monkeypatch):
    import copy
    from dataclasses import replace
    from mrta_ranker.features import extract_c2_features,extract_c4_features
    from mrta_search.lns import atomic_complete_candidate
    import mrta_search.direction as direction_module
    import mrta_reference.scheduler as scheduler_module
    def forbidden(*args,**kwargs): raise AssertionError('Feature extraction invoked expensive evaluator')
    monkeypatch.setattr(scheduler_module,'reference_schedule',forbidden)
    monkeypatch.setattr(direction_module,'optimize_directions_with_initial_feasibility',forbidden)
    current=_base();directions=((),(),(0,0),(0,0))
    candidate=atomic_complete_candidate(current,directions,_family_candidates()[0],FAST)
    candidate=replace(candidate,projected_process_makespan=6.0,local_directed_travel_delta=0.0)
    direction=ConstrainedDirectionResult(DirectionStatus.FEASIBLE,directions,5.0)
    before=copy.deepcopy((current,directions,candidate,direction));rng=random.getstate()
    args=(current,directions,100.0,candidate,FAST)
    c2=extract_c2_features(*args)
    c4=extract_c4_features(*args,direction=direction,cheap_rank=2)
    assert len(c2)==87 and len(c4)==130
    assert c2==extract_c2_features(*args)
    assert c4==extract_c4_features(*args,direction=direction,cheap_rank=2)
    assert before==(current,directions,candidate,direction) and rng==random.getstate()
    assert not any(word in key for key in c2 for word in ('direction','reference','wait','deadlock','certif','schedule','c3','c4'))
    assert not any(word in key for key in c4 for word in ('reference','wait','deadlock','certif','schedule'))
    assert c2.items()<=c4.items()
    assert c2['before_max_robot_load']==6.0 and c2['travel_proxy_before']==5.0
    assert c4['direction_dp_travel']==5.0 and c4['direction_flips']==0.0
    assert c4['direction_r2_first_x']==1.0 and c4['direction_r2_last_x']==5.0
    unavailable=ConstrainedDirectionResult(DirectionStatus.NO_LEGAL_FIRST_ORIENTATION,((),(),(),()),None)
    masked=extract_c4_features(*args,direction=unavailable,cheap_rank=2)
    assert tuple(masked)==tuple(c4)
    assert masked['direction_cost_available']==0.0 and masked['direction_r2_route_available']==0.0
    import math
    assert all(math.isfinite(value) for row in (c2,c4,masked) for value in row.values())
    with pytest.raises(ValueError): extract_c2_features(current,directions,0.0,candidate,FAST)


def test_ranker_targets_exclude_numeric_and_leave_deadlock_regression_missing():
    from mrta_ranker.features import make_training_targets
    assert make_training_targets('NUMERIC_FAILURE',100.0,None) is None
    assert make_training_targets('DEADLOCK',100.0,None)=={'y_feasible':0,'y_improvement':None}
    assert make_training_targets('DIRECTION_INFEASIBLE',100.0,None)=={'y_feasible':0,'y_improvement':None}
    assert make_training_targets('FEASIBLE_CERTIFIED',100.0,90.0)=={'y_feasible':1,'y_improvement':.1}
    assert make_training_targets('FEASIBLE_CERTIFIED',100.0,110.0)=={'y_feasible':1,'y_improvement':-.1}
    with pytest.raises(ValueError): make_training_targets('FEASIBLE_CERTIFIED',100.0,None)
    with pytest.raises(ValueError): make_training_targets('FEASIBLE_CERTIFIED',0.0,10.0)
    with pytest.raises(ValueError): make_training_targets('typo',100.0,None)


def test_v2_production_pool_keeps_historical_explicit_configs(monkeypatch):
    from dataclasses import asdict
    from mrta_search import pipeline
    calls=[];marker=object()
    def capture(parents,config,search_config,**kwargs):
        calls.append(search_config)
        return marker
    monkeypatch.setattr(pipeline,'run_bounded_sa_oi',capture)
    assert pipeline.run_sa_oi_alns_v2(()) is marker
    production=calls[-1]
    assert production.m==pipeline.SA_OI_ALNS_V2_PRODUCTION_POOL_SIZE==192
    assert production.m_lns==48 and production.m_atomic is None
    assert (production.kdp,production.kref,production.kref_total)==(8,2,4)
    historical=SearchConfig()
    assert historical.m==64 and historical.m_lns==16
    assert {k:v for k,v in asdict(production).items() if k not in ('m','m_lns')}=={k:v for k,v in asdict(historical).items() if k not in ('m','m_lns')}
    pipeline.run_sa_oi_alns_v2((),search_config=historical)
    assert calls[-1] is historical
