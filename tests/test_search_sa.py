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
from scripts.run_phase3_baseline_smoke import phase3_alns_config


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


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

# Phase4-0 observation/replay is tested in this existing search test module.
def test_phase4_observer_and_prefix_replay_leave_search_unchanged(monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    from scripts import run_phase4_0_oracle_recall as audit
    from mrta_search import pipeline
    from mrta_search.neighborhood import generate_raw_attempts
    from mrta_reference.scope import FORMAL_SCOPE_V2
    parents=_base().parents
    cfg=replace(audit.SEARCH,max_iterations=3,time_limit=None)
    captures=[]
    original=pipeline.evaluate_iteration
    def observed_evaluation(*args,**kwargs):
        result=original(*args,**kwargs)
        captures.append((result.complete_candidates,result.c3_candidates,result.c4_candidates))
        return result
    monkeypatch.setattr(pipeline,'evaluate_iteration',observed_evaluation)
    a=pipeline.run_sa_oi_alns_v2(parents,audit.CONFIG,cfg,seed=19,source_provenance=_development_provenance())
    off=captures[:];captures.clear()
    cap=audit.Capture(keep_all=True)
    b=pipeline.run_sa_oi_alns_v2(parents,audit.CONFIG,cfg,seed=19,source_provenance=_development_provenance(),observer=cap)
    assert captures==off
    assert a.best_solution==b.best_solution and a.best_directions==b.best_directions
    assert a.best_metrics==b.best_metrics
    assert a.stats.proposal_trajectory==b.stats.proposal_trajectory
    assert a.stats.operator_sequence==b.stats.operator_sequence
    before=deepcopy(b.stats)
    for context in cap.all[:-1]:
        saved=deepcopy(context)
        candidates,trace,chunks,c2,c3,c4=audit.replay(context,seed=19)
        assert len(trace)==256 and trace[:64]==context['attempts']
        assert chunks[0][0]==64 and chunks[1][0]==128 and chunks[2][0]==256
        assert len(c2)<=8 and len(c4)<=2
        assert len(candidates)<256  # rejected/duplicate attempts consume budget
        assert context==saved
        assert audit.unpack(audit.pack(context,parents),parents)==context
        generation_seed=19+context['iteration']*65537
        actual_atomic=generate_raw_attempts(context['solution'],audit.CONFIG,m=192,seed=generation_seed,
            stats=SearchStats(FORMAL_SCOPE_V2.scope_id,19),enable_x_split=True,scope=FORMAL_SCOPE_V2)
        assert actual_atomic==tuple(v[0] for event,v in trace if event=='atomic_attempt')
        lns=[]
        pipeline.generate_candidate_pool(context['solution'],context['directions'],audit.CONFIG,audit.SEARCH,
            SearchStats(FORMAL_SCOPE_V2.scope_id,19),seed=generation_seed,current_schedule=context['schedule'],
            adaptive_state=deepcopy(context['adaptive']),scope=FORMAL_SCOPE_V2,enable_x_split=True,
            atomic_budget=0,lns_budget=64,observer=lambda event,value:lns.append((event,value)))
        assert [v[:4] for event,v in lns]==[v[:4] for event,v in trace if event=='lns_attempt']
    assert b.stats==before


def test_phase4_recall_nulls_epsilon_and_nested_minima():
    from scripts.run_phase4_0_oracle_recall import recall_metrics
    def c(key,prefix,cmax,status='FEASIBLE_CERTIFIED'):
        return dict(candidate_id=key,first_prefix=prefix,Cmax=cmax,status=status)
    rows=[c('a',64,90),c('b',128,80),c('c',256,60),c('dead',64,None,'DEADLOCK')]
    m=recall_metrics(100,rows,['a'],['a'])
    assert m['C_star']=={'64':90,'128':80,'256':60,'C2':90,'C4':90}
    assert m['generation_recall64']==.25 and m['generation_recall128']==.5
    assert m['c2_recall']==1 and m['c4_recall']==1
    m=recall_metrics(100,[c('dead',64,None,'DEADLOCK'),c('b',128,80)],[],[])
    assert m['C_star']['64'] is None and m['no_feasible']['64']
    assert m['zero_capture64'] and m['generation_recall64']==0
    assert m['c2_recall'] is None and m['c4_recall'] is None
    m=recall_metrics(100,[c('a',64,100-5e-8)],['a'],[])
    assert not m['opportunity'] and m['generation_recall64'] is None
    m=recall_metrics(100,[c('a',64,90)],[],[])
    assert m['c2_recall']==0 and m['c4_recall']==0 and m['zero_captureC4']


@pytest.mark.parametrize('status',('DEADLOCK','INFEASIBLE','NUMERIC_FAILURE','DIRECTION_INFEASIBLE','FEASIBLE','CERTIFIER_MISMATCH'))
def test_phase4_label_semantics(monkeypatch,status):
    from types import SimpleNamespace
    from scripts import run_phase4_0_oracle_recall as audit
    from mrta_reference.model import OfficialMetrics
    solution=_base()
    candidate=SimpleNamespace(solution=solution)
    direction=SimpleNamespace(total_empty_travel=None if status=='DIRECTION_INFEASIBLE' else 1.0,directions=((),(),(0,0),(0,0)))
    monkeypatch.setattr(audit,'canonicalize',lambda *a,**kw:solution)
    monkeypatch.setattr(audit,'optimize_directions_with_initial_feasibility',lambda *a:direction)
    monkeypatch.setattr(audit,'encode',lambda value,*a:repr(value))
    monkeypatch.setattr(audit,'identity',lambda c:'id')
    monkeypatch.setattr(audit,'certify_schedule',lambda *a,**kw:SimpleNamespace(certified=status!='CERTIFIER_MISMATCH'))
    calls=[]
    def evaluator(*args,**kwargs):
        calls.append(1)
        schedule_status=ScheduleStatus.FEASIBLE if status in ('FEASIBLE','CERTIFIER_MISMATCH') else ScheduleStatus(status)
        return ScheduleResult(schedule_status,cmax=90.0 if schedule_status is ScheduleStatus.FEASIBLE else None)
    label=audit.evaluate_oracle_candidate(candidate,100,evaluator)
    if status=='FEASIBLE':
        assert label[:4]==('FEASIBLE_CERTIFIED',90.0,10.0,1)
    elif status=='CERTIFIER_MISMATCH':
        assert label[0]=='NUMERIC_FAILURE' and label[1:3]==(None,None) and label[8]==1
    else:
        assert label[0]==status and label[1:3]==(None,None)
    assert len(calls)==(0 if status=='DIRECTION_INFEASIBLE' else 1)


@pytest.mark.parametrize('role',('RANKER_TRAIN','V2_VALIDATION','ID_TEST_SEALED','V2_MODEL_DEVELOPMENT_CONSUMED'))
def test_phase4_isolation_fails_before_excel(monkeypatch,role):
    from scripts import run_phase4_0_oracle_recall as audit
    calls=[]
    original=audit.read
    parent_role='V2_TRAIN_POOL' if role=='RANKER_TRAIN' else role
    def read(path):
        if path==audit.ROLES:
            return {'workbooks':[{'relative_path':'blocked.xlsx','new_v2_role':parent_role}]}
        if path==audit.SPLIT:
            return {'workbooks':[{'relative_path':'blocked.xlsx','ranker_role':role}]}
        return original(path)
    monkeypatch.setattr(audit,'read',read)
    monkeypatch.setattr(audit,'load_ppo_platform_instance',lambda *a,**kw:calls.append(1))
    with pytest.raises(ValueError):
        audit.load_parents({'relative_path':'blocked.xlsx'})
    assert not calls


def test_phase4_mechanical_bottleneck_priority():
    from scripts.run_phase4_0_oracle_recall import decisions,recall_metrics
    def rows(c64,c128,c256,c2,c4):
        result=[]
        for tier in ('SMALL','MEDIUM','LARGE'):
            for _ in range(6):
                candidates=[dict(candidate_id=k,first_prefix=p,Cmax=v,status='FEASIBLE_CERTIFIED') for k,p,v in [('base',64,c64),('two',64,c2),('four',64,c4),('b',128,c128),('c',256,c256)]]
                result.append({'tier':tier,**recall_metrics(100,candidates,['two','four'],['four'])})
        return result
    assert decisions(rows(99,80,80,100,100),True)['PRIMARY_BOTTLENECK']=='CANDIDATE_GENERATION'
    assert decisions(rows(80,80,80,90,95),True)['PHASE4_1_MLP_AUTHORIZED']=='YES'
    assert decisions(rows(80,80,80,80,80),True)['PRIMARY_BOTTLENECK']=='NO_MATERIAL_POOL_OR_RANKING_GAP'
    assert decisions(rows(100,100,100,100,100),True)['PRIMARY_BOTTLENECK']=='INSUFFICIENT_EVIDENCE'
    assert decisions(rows(80,80,80,90,95),False)['PHASE4_1_MLP_AUTHORIZED']=='NO'


def test_phase4_offline_worker_preserves_label_and_diagnostics():
    from scripts import run_phase4_0_oracle_recall as audit
    from mrta_search.lns import atomic_complete_candidate
    from mrta_reference.scheduler import FormalReferenceEvaluator
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.neighborhood import generate_raw_attempts, screen_raw_attempts
    current=_base()
    directions=((),(),(0,0),(0,0))
    stats=SearchStats(FORMAL_SCOPE_V2.scope_id,19)
    raw=generate_raw_attempts(current,audit.CONFIG,m=48,seed=19,stats=stats,enable_x_split=True,scope=FORMAL_SCOPE_V2)
    screened=screen_raw_attempts(current,directions,raw,audit.CONFIG,stats,enable_x_split=True,scope=FORMAL_SCOPE_V2)
    candidate=atomic_complete_candidate(current,directions,screened[0],audit.CONFIG)
    expected=audit.evaluate_oracle_candidate(candidate,1000,FormalReferenceEvaluator(FORMAL_SCOPE_V2))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as executor:
        actual=executor.submit(audit.oracle_worker,(audit.pack(candidate,current.parents),current.parents,1000,())).result(timeout=30)
    assert actual[:4]==expected[:4]
    assert actual[8]==expected[8]
    assert bytes(actual[-1])==bytes(expected[-1])


@pytest.mark.parametrize('column',('protocol_hash','scope_hash','source_hash'))
def test_phase4_resume_rejects_mixed_scientific_identity(monkeypatch,column):
    import sqlite3
    from scripts import run_phase4_0_oracle_recall as audit
    connection=sqlite3.connect(':memory:')
    monkeypatch.setattr(audit.sqlite3,'connect',lambda *args,**kwargs:connection)
    protocol={'protocol_hash':'protocol','scope_hash':audit.FORMAL_SCOPE_V2.scope_hash,
              'source':{'source_tree_hash':'source'}}
    db=audit.database(protocol)
    db.execute('INSERT INTO states VALUES (?,?,?)',('state','{}','PASS'))
    db.execute('INSERT INTO candidates(protocol_hash,state_id,candidate_id,scope_hash,source_hash,canonical_hash,family,first_prefix,production_reference_evaluated,payload) VALUES (?,?,?,?,?,?,?,?,?,?)',
        ('protocol','state','candidate',protocol['scope_hash'],'source','canonical','STRUCTURAL',64,0,b'payload'))
    db.execute('UPDATE candidates SET '+column+'=?',('wrong',))
    db.commit()
    with pytest.raises(ValueError,match='different protocol/scope/source identity'):
        audit.database(protocol)
    db.close()


def test_phase4_split_uses_only_whole_training_workbooks():
    from collections import Counter
    from scripts import run_phase4_0_oracle_recall as audit
    split=audit.read(audit.SPLIT)
    paths={row['relative_path'] for row in split['workbooks']}
    assert len(paths)==31
    assert Counter(row['ranker_role'] for row in split['workbooks'])=={'RANKER_TRAIN':23,'RANKER_DEV':8}
    roles={row['relative_path']:row['new_v2_role'] for row in audit.read(audit.ROLES)['workbooks']}
    assert all(roles[path]=='V2_TRAIN_POOL' for path in paths)
    counts=Counter(row['relative_path'] for row in split['audit_instances'])
    assert max(counts.values())<=2
    assigned={row['relative_path']:row['ranker_role'] for row in split['workbooks']}
    assert all(assigned[row['relative_path']]=='RANKER_DEV' for row in split['audit_instances'])
    assert Counter(row['tier'] for row in split['audit_instances'])=={'SMALL':4,'MEDIUM':4,'LARGE':4}


def test_phase4_storage_compaction_preserves_every_logical_row(monkeypatch):
    import sqlite3
    from scripts import run_phase4_0_oracle_recall as audit
    connection=sqlite3.connect(':memory:')
    monkeypatch.setattr(audit.sqlite3,'connect',lambda *args,**kwargs:connection)
    protocol={'protocol_hash':'protocol','scope_hash':audit.FORMAL_SCOPE_V2.scope_hash,
              'source':{'source_tree_hash':'source'}}
    db=audit.database(protocol)
    expected={}
    for state in ('first','second'):
        db.execute('INSERT INTO states VALUES (?,?,?)',(state,'{}','PASS'))
        db.execute('INSERT INTO candidates(protocol_hash,state_id,candidate_id,scope_hash,source_hash,canonical_hash,family,first_prefix,production_reference_evaluated,payload,status,Cmax,delta_Cmax) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            ('protocol',state,'完整候选身份:long identity'*20,protocol['scope_hash'],'source','canonical','STRUCTURAL',64,0,audit.pack({'candidate':'payload'}),'FEASIBLE_CERTIFIED',90,10))
        for ordinal in range(4):
            detail=('atomic_attempt',{'ordinal':ordinal,'shared':'a repeated trace value'*20})
            expected[state,ordinal]=detail
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(state,ordinal,'ATOMIC','VALID','STRUCTURAL','完整候选身份:long identity'*20,audit.pack(detail)))
    import zlib
    diagnostic={'certified':True,'note':'完整诊断','shared':'repeated data'*20}
    db.execute('UPDATE candidates SET diagnostics=?',(zlib.compress(audit.dumps(diagnostic).encode('utf-8')),))
    db.commit()
    candidate_id='完整候选身份:long identity'*20
    assert audit.read_candidate_payload(db,'first',candidate_id)=={'candidate':'payload'}
    assert audit.read_candidate_diagnostic(db,'first',candidate_id)==diagnostic
    assert audit.compact_connection(db)
    assert audit.compact_connection(db)  # repeated compaction is lossless too
    assert audit.read_candidate_payload(db,'first',candidate_id)=={'candidate':'payload'}
    assert audit.read_candidate_diagnostic(db,'first',candidate_id)==diagnostic
    for key,value in expected.items():
        assert audit.read_attempt_detail(db,*key)==value
    identity=db.execute('SELECT candidate_id FROM candidates LIMIT 1').fetchone()[0]
    assert isinstance(identity,bytes)
    assert audit.expanded_identity(identity)=='完整候选身份:long identity'*20
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    with pytest.raises(sqlite3.IntegrityError):
        db.execute('INSERT INTO candidates SELECT * FROM candidates LIMIT 1')
    db.close()

def test_phase4b_balanced_prefixes_and_duplicate_promotion():
    from collections import Counter
    from scripts import analyze_phase4_0b_pool_scaling as b
    attempts=[]
    ai=li=0
    for atomic,lns in ((48,16),(48,16),(96,32)):
        for _ in range(atomic):
            attempts.append(dict(ordinal=len(attempts),source='ATOMIC',status='VALID',canonical_hash=f'a{ai}'))
            ai+=1
        for _ in range(lns):
            # LNS #16 duplicated atomic #90 in the full original chunk. At
            # M80, atomic #90 is absent, so the frozen LNS payload is valid.
            canonical='a90' if li==16 else f'l{li}'
            attempts.append(dict(ordinal=len(attempts),source='LNS_REPAIRED',status='DUPLICATE' if li==16 else 'VALID',canonical_hash=canonical))
            li+=1
    attempts[0].update(status='IDENTITY',canonical_hash=None)
    attempts[1].update(status='CONSTRUCTION_REJECTED',canonical_hash=None)
    labels={a['canonical_hash']:{'status':'FEASIBLE_CERTIFIED','Cmax':80} for a in attempts if a['canonical_hash'] is not None}
    previous=set()
    for m in b.SIZES:
        chosen=b.select_attempt_prefix(attempts,m)
        assert Counter(a['source'] for a in chosen)=={'ATOMIC':3*m//4,'LNS_REPAIRED':m//4}
        result=b.build_prefix(chosen,labels)
        assert previous<=set(result['representatives'])
        previous=set(result['representatives'])
        assert sum(result['counts'].values())==m
        if m in (64,128,256):
            original=b.build_prefix(attempts[:m],labels)
            assert result['counts']==original['counts']
            assert set(result['representatives'])==set(original['representatives'])
    prefix80=b.build_prefix(b.select_attempt_prefix(attempts,80),labels)
    assert prefix80['promoted_duplicates']==1
    assert prefix80['representatives']['a90']['source']=='LNS_REPAIRED'
    assert prefix80['counts']['IDENTITY']==1
    assert prefix80['counts']['CONSTRUCTION_REJECTED']==1
    assert len(b.select_attempt_prefix(attempts,80))==80  # never refill


def test_phase4b_materiality_nulls_and_literal_signed_recall(tmp_path,monkeypatch):
    from scripts import analyze_phase4_0b_pool_scaling as b
    assert not b.in_materiality(None,0)
    assert not b.in_materiality(0,0) and b.in_materiality(1e-12,0)
    for threshold in (.001,.005,.01):
        assert b.in_materiality(threshold,threshold)
        assert not b.in_materiality(threshold-1e-12,threshold)
    null=b.generation_metrics(100,None,None)
    assert null['C_star'] is None and null['normalized_generation_regret'] is None
    assert null['generation_recall'] is None and not null['opportunity']
    missing=b.generation_metrics(100,None,90)
    assert missing['normalized_generation_regret'] is None
    assert missing['selection_regret_with_incumbent_fallback']==pytest.approx(.1)
    assert missing['generation_recall']==0 and missing['zero_capture']
    no_gain=b.generation_metrics(100,100,100)
    assert no_gain['generation_recall'] is None
    signed=b.screening_metrics(100,90,110)
    assert signed['recall']==-1 and signed['capture_recall']==0
    assert signed['normalized_regret']==.2 and signed['incumbent_fallback_regret']==.1
    selected_null=b.screening_metrics(100,90,None)
    assert selected_null['recall'] is None and selected_null['normalized_regret'] is None
    assert selected_null['zero_capture'] and selected_null['capture_recall']==0
    assert b.quantile([0,10],.9)==9
    # The historical writer must keep its original whitelist; new approved
    # analysis outputs belong to the analysis writer, not that legacy API.
    import json
    monkeypatch.setattr(b,'OUTPUT',tmp_path/'analysis.json')
    monkeypatch.setattr(b,'REPORT',tmp_path/'analysis.md')
    monkeypatch.setattr(b,'report_text',lambda result:'# 已完成分析\n')
    monkeypatch.setattr(b.audit,'write',lambda *args:pytest.fail('Historical writer must not be used'))
    b.write_outputs({'recommended_pool_size':192,'中文':'无损'})
    assert json.loads(b.OUTPUT.read_text(encoding='utf-8'))['recommended_pool_size']==192
    assert b.REPORT.read_text(encoding='utf-8')=='# 已完成分析\n'


def test_phase4b_minimum_global_pool_selector_exact_boundaries():
    from copy import deepcopy
    from scripts import analyze_phase4_0b_pool_scaling as b
    states=[]
    for i in range(10):
        row={'selection_regret_with_incumbent_fallback':.001 if i<8 else .005,
             'oracle_improvement_ratio':.005,
             'generation_recall':.8 if i<8 else .1 if i==8 else 0,
             'zero_capture':i==9}
        states.append({'prefixes':{'64':row}})
    passed=b.selection_criteria(states,64)
    assert passed['satisfies'] and all(passed['conditions'].values())
    variants=[]
    worse=deepcopy(states);worse[9]['prefixes']['64']['selection_regret_with_incumbent_fallback']+=1e-8;variants.append(worse)
    worse=deepcopy(states);worse[0]['prefixes']['64']['generation_recall']=.8-1e-8;variants.append(worse)
    worse=deepcopy(states);worse[8]['prefixes']['64']['zero_capture']=True;variants.append(worse)
    for variant in variants:
        assert not b.selection_criteria(variant,64)['satisfies']
    criteria={str(m):{'satisfies':False} for m in b.SIZES}
    assert b.choose_pool(criteria) is None
    criteria['80']=passed
    assert b.choose_pool(criteria)==80
    criteria['64']=passed
    assert b.choose_pool(criteria)==64


def test_phase4b_replay_calls_real_policy_with_stored_directions():
    from types import SimpleNamespace
    from scripts import analyze_phase4_0b_pool_scaling as b
    from mrta_search.lns import atomic_complete_candidate
    from mrta_search.pipeline import select_c2_by_family,select_c4_by_family
    current=_base();dirs=((),(),(0,0),(0,0))
    pool=[atomic_complete_candidate(current,dirs,c,FAST) for c in _family_candidates()]
    calls=[]
    def direction(canonical):
        calls.append(canonical)
        return SimpleNamespace(total_empty_travel=10.0)
    labels={current.canonical_hash:{'status':'FEASIBLE_CERTIFIED','Cmax':80}}
    actual=b.replay_screening(pool,labels,direction,17,3,100)
    ranked=sorted(pool,key=lambda c:c.cheap_score)
    expected2=select_c2_by_family(ranked,8,seed=17,iteration=3)
    expected3=[DirectionEvaluatedCandidate(c,ranked.index(c),SimpleNamespace(total_empty_travel=10.0)) for c in expected2]
    expected4=select_c4_by_family(expected3,2,seed=17,iteration=3)
    assert actual['C2_ids']==[b.audit.identity(c) for c in expected2]
    assert actual['C4_ids']==[b.audit.identity(c.screened) for c in expected4]
    assert len(calls)==8 and len(actual['C4_ids'])==2


def test_phase4b_existing_artifact_prefixes_and_screening_without_solver(monkeypatch):
    import sqlite3
    from scripts import analyze_phase4_0b_pool_scaling as b
    a=b.audit
    def forbidden(*args,**kwargs):
        raise AssertionError('Re-analysis must not run a solver or generator')
    for name in ('generate_candidate_pool','optimize_directions_with_initial_feasibility','evaluate_oracle_candidate','FormalReferenceEvaluator','replay'):
        monkeypatch.setattr(a,name,forbidden)
    data=a.read(a.CONTEXTS);original=a.read(a.RESULT)
    row=data['contexts'][0]
    db=sqlite3.connect('file:'+a.LABELS.as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    meta,context,labels,attempts,prefixes,repaired,cache=b.load_state(db,row,data)
    old=next(s for s in original['states'] if s['state_context_id']==row['state_context_id'])
    for m in (64,128,256):
        assert prefixes[str(m)]['C_star']==old['C_star'][str(m)]
        assert prefixes[str(m)]['valid_unique']==old['unique_counts'][str(m)]
    pool=[a.read_candidate_payload(db,meta['state_context_id'],v['candidate_id'],context['solution'].parents,packet_cache=cache) for v in prefixes['64']['representatives'].values()]
    def direction(canonical):
        return a.decode(a.read_candidate_diagnostic(db,meta['state_context_id'],labels[canonical]['candidate_id'],packet_cache=cache)['direction'])
    result=b.replay_screening(pool,labels,direction,meta['seed'],meta['iteration'],meta['C_s'])
    assert result['C2_ids']==meta['C2'] and result['C4_ids']==meta['C4']
    # Cached and uncached readers expand precisely the same existing record.
    cid=next(iter(labels.values()))['candidate_id']
    assert a.read_candidate_payload(db,meta['state_context_id'],cid,context['solution'].parents,packet_cache=cache)==a.read_candidate_payload(db,meta['state_context_id'],cid,context['solution'].parents)
    with pytest.raises(sqlite3.OperationalError,match='readonly'):
        db.execute('DELETE FROM candidates')
    db.close()

def test_phase4_1a_configs_only_change_attempt_budgets():
    from dataclasses import asdict
    from scripts import run_phase4_1a_production_pool as smoke
    historical=asdict(smoke.audit.SEARCH)
    for m in smoke.SIZES:
        cfg=asdict(smoke.config_for(m))
        assert cfg.pop('m')==m
        assert cfg.pop('m_lns')==m//4
        assert m-m//4==3*m//4
        assert cfg=={k:v for k,v in historical.items() if k not in ('m','m_lns')}
        assert (cfg['kdp'],cfg['kref'],cfg['kref_total'])==(8,2,4)
    with pytest.raises(ValueError): smoke.config_for(256)


def test_phase4_1a_checkpoint_excludes_overshoot_and_initialization_late():
    from types import SimpleNamespace
    from scripts.run_phase4_1a_production_pool import checkpoint_cmax
    stats=SearchStats(scope_id="V2_TEST",seed=0)
    stats.record_best(4.0,100.0)
    stats.record_best(60.0,90.0)
    stats.record_best(60.001,70.0)
    result=SimpleNamespace(stats=stats,anytime=stats.anytime((60.0,)))
    assert checkpoint_cmax(result)==90.0
    stats.best_events=[(60.001,100.0)]
    result.anytime=stats.anytime((60.0,))
    assert checkpoint_cmax(result) is None


def _phase4_1a_synthetic_runs():
    from scripts.run_phase4_1a_production_pool import SIZES,TIERS
    rows=[]
    for index in range(18):
        for m in SIZES:
            row={'instance_id':str(index//3),'seed':index%3,'tier':TIERS[index//6],
                 'M':m,'checkpoint_certified':True,'Cmax_at_60':100.0,
                 'iterations_completed':100 if m==64 else 40}
            for name in ('candidate_generation_time_fraction','reference_calls','reference_calls_per_second',
                'candidate_generation_seconds','lns_seconds','direction_dp_seconds','reference_scheduler_seconds',
                'valid_unique_candidates','unique_candidates_per_second','actual_runtime','overshoot','initialization_seconds'):
                row[name]=1.0
            rows.append(row)
    return rows


def test_phase4_1a_selects_smallest_pool_and_respects_boundaries():
    from scripts.run_phase4_1a_production_pool import compare_and_choose
    rows=_phase4_1a_synthetic_runs()
    assert compare_and_choose(rows)[0]==64
    # 64 violates the overall threshold; 128 sits on the inclusive limits.
    for r in rows:
        if r['M']==64: r['Cmax_at_60']=102.0
        if r['M']==128: r['Cmax_at_60']=101.0
    chosen,summary=compare_and_choose(rows)
    assert chosen==128 and summary['128']['median_iteration_retention_vs_M64']==.4
    for r in rows:
        if r['M']==128: r['iterations_completed']=39
    assert compare_and_choose(rows)[0]==160
    # A tier median above 2% rejects an otherwise good overall median.
    for r in rows:
        if r['M']==160 and r['tier']=='LARGE': r['Cmax_at_60']=102.01
    assert compare_and_choose(rows)[0]==192
    next(r for r in rows if r['M']==192)['checkpoint_certified']=False
    chosen,summary=compare_and_choose(rows)
    assert chosen==64 and not any(s['satisfies'] for s in summary.values())
    with pytest.raises(ValueError): compare_and_choose(rows+[rows[0]])


def test_phase4_1a_workbook_split_is_geometry_only_and_isolated(monkeypatch):
    from scripts import run_phase4_1a_production_pool as smoke
    def forbidden(*args,**kwargs): raise AssertionError('Solver/label access during split')
    monkeypatch.setattr(smoke.audit,'load_parents',forbidden)
    monkeypatch.setattr(smoke.audit,'read_candidate_payload',forbidden)
    split=smoke.compute_mlp_split()
    assert split==smoke.compute_mlp_split()
    train,dev,consumed=(set(split[k]) for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED'))
    assert (len(train),len(dev),len(consumed))==(17,6,8)
    assert not (train&dev or train&consumed or dev&consumed)
    assert not {case['relative_path'] for case in smoke.development_cases()} & (train|dev|consumed)
    old=smoke.audit.read(smoke.audit.SPLIT)['workbooks']
    assert train|dev=={w['relative_path'] for w in old if w['ranker_role']=='RANKER_TRAIN'}
    assert consumed=={w['relative_path'] for w in old if w['ranker_role']=='RANKER_DEV'}


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


def test_phase4_1a_timing_wrapper_preserves_values_rng_and_restores(monkeypatch):
    from scripts import run_phase4_1a_production_pool as smoke
    marker=object()
    def destroy(): return marker
    def repair(value): return value
    def generate(): return smoke.pipeline.repair_partial_state(smoke.pipeline.destroy_parents())
    for name,fn in (('destroy_parents',destroy),('repair_partial_state',repair),('generate_candidate_pool',generate)):
        monkeypatch.setattr(smoke.pipeline,name,fn)
    ticks=iter(range(6))
    monkeypatch.setattr(smoke.time,'perf_counter',lambda:next(ticks))
    rng=random.getstate()
    with smoke.generation_timers() as totals:
        assert smoke.pipeline.generate_candidate_pool() is marker
    assert totals=={'candidate_generation_seconds':5.0,'lns_seconds':2.0}
    assert random.getstate()==rng
    assert smoke.pipeline.generate_candidate_pool is generate
    assert smoke.pipeline.destroy_parents is destroy
    assert smoke.pipeline.repair_partial_state is repair


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
