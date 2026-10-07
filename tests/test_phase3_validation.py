import copy,random,sqlite3
from pathlib import Path
import pytest
from mrta_ranker import dataset as d


def test_mlp_static_representatives_and_workbook_whitelist(monkeypatch):
    split=d.validate_split(d.read(d.SPLIT));entries=d.representatives(split)
    assert entries==d.representatives(split)
    assert len({e['relative_path'] for e in entries})==23
    for role,counts in (('MLP_TRAIN',(6,6,5)),('MLP_DEV',(2,2,2))):
        rows=[e for e in entries if e['split']==role]
        assert tuple(sum(e['target_tier']==t for e in rows) for t in d.TIERS)==counts
        assert {e['relative_path'] for e in rows}==set(split[role])
    called=[]
    for role,path in [('ORACLE_DEV_CONSUMED',split['ORACLE_DEV_CONSUMED'][0]),('MLP_TRAIN','forbidden.xlsx')]:
        with pytest.raises(PermissionError):
            d.load_parents({'split':role,'relative_path':path},loader=lambda *a,**k:called.append(1))
    assert not called


def test_accepted_capture_never_uses_future_boundary():
    cap=d.Capture()
    for iteration,elapsed in enumerate((1.0,4.0,7.0,29.0,31.0,59.0,61.0)):
        cap('boundary',{'iteration':iteration,'elapsed':elapsed,'adaptive':{'weight':[1]}})
    assert {k:v['elapsed'] for k,v in cap.snapshots.items()}=={'EARLY':4.,'MID':29.,'LATE':59.}
    cap=d.Capture();cap('boundary',{'iteration':0,'elapsed':6.,'adaptive':{}})
    assert 'EARLY' not in cap.snapshots and cap.first_boundary_elapsed==6.0


def test_current_dataset_config_and_typed_state_roundtrip():
    from test_search_sa import _base
    cfg=d.search_config()
    assert (cfg.m,cfg.m_lns,cfg.m_atomic,cfg.kdp,cfg.kref,cfg.kref_total)==(192,48,None,8,2,4)
    assert cfg.time_limit==60 and not cfg.enable_two_opt_star
    solution=_base();parents=solution.parents
    assert d.unpack(d.pack(solution,parents),parents)==solution


def test_learned_family_selectors_match_production_when_scores_match():
    from test_search_sa import _family_candidates
    from mrta_ranker.ranking import family_c2,family_c4
    from mrta_search.pipeline import decision_family,select_c2_by_family,select_c4_by_family,DirectionEvaluatedCandidate
    from mrta_search.direction import ConstrainedDirectionResult,DirectionStatus
    candidates=_family_candidates()
    directed=[DirectionEvaluatedCandidate(c,i,ConstrainedDirectionResult(DirectionStatus.FEASIBLE,((),(),(),()),1.)) for i,c in enumerate(candidates)]
    for seed in range(4):
        assert family_c2(candidates,lambda c:c.cheap_score,decision_family,seed=seed)==select_c2_by_family(candidates,8,seed=seed,iteration=0)
        assert family_c4(directed,lambda c:c.rerank_key,lambda c:decision_family(c.screened),seed=seed)==select_c4_by_family(directed,2,seed=seed,iteration=0)
    # The real rotated fallback is retained when TARGET_X is absent.
    subset=directed[:12]+directed[-3:-1]
    assert family_c4(subset,lambda c:c.rerank_key,lambda c:decision_family(c.screened),seed=3)==select_c4_by_family(subset,2,seed=3,iteration=0)



def test_ranker_train_only_scaling_and_masked_regression(tmp_path):
    torch=pytest.importorskip('torch')
    import numpy as np
    from mrta_ranker.model import fit_scaler,transform,CandidateMLP,multitask_loss,predict_scores,save_model,load_model
    train=np.array([[1.,3.],[3.,3.]])
    mean,std=fit_scaler(train)
    assert mean.tolist()==[2.,3.] and std.tolist()==[1.,1.]
    assert transform([[102.,3.]],mean,std).tolist()==[[100.,0.]]
    assert mean.tolist()==[2.,3.]  # DEV apply cannot refit TRAIN statistics.
    assert np.array_equal(transform([[1.00000001,3.]],mean,std),transform(np.array([[1.00000001,3.]],dtype=np.float32),mean,std))
    logit=torch.tensor([0.,0.],requires_grad=True);gain=torch.tensor([.1,999.],requires_grad=True)
    loss,bce,reg=multitask_loss(logit,gain,torch.tensor([1.,0.]),torch.tensor([.2,float('nan')]))
    assert torch.isfinite(loss) and float(reg.detach())==pytest.approx(.005)
    loss.backward();assert gain.grad[1]==0
    loss,_,reg=multitask_loss(logit,gain,torch.zeros(2),torch.full((2,),float('nan')))
    assert torch.isfinite(loss) and reg==0
    model=CandidateMLP(2)
    with torch.no_grad():
        for param in model.parameters(): param.zero_()
        model.improvement.bias.fill_(-.2)
    assert predict_scores(model,train,mean,std).tolist()==pytest.approx([-.1,-.1])
    path=tmp_path/'ranker.pt';save_model(path,model,'C2',['a','b'],1,{'ALL':{}})
    restored,meta=load_model(path)
    assert meta['width']==2 and meta['stage']=='C2'
    assert predict_scores(restored,train,mean,std).tolist()==pytest.approx([-.1,-.1])


def test_native_ranker_hook_preserves_budget_and_tie_policy():
    from test_search_sa import _base,FAST,_development_provenance
    from mrta_search import pipeline
    from mrta_ranker.ranking import LearnedRanker
    from dataclasses import replace
    ranker=LearnedRanker.__new__(LearnedRanker)
    ranker.models={'C2':None,'C4':None};ranker.inference_seconds=0.0
    ranker.scores=lambda stage,features:[0.0]*len(features)
    cfg=replace(d.search_config(),max_iterations=1,time_limit=None)
    kwargs={'seed':20261121,'source_provenance':_development_provenance()}
    baseline=pipeline.run_sa_oi_alns_v2(_base().parents,FAST,cfg,**kwargs)
    learned=pipeline.run_sa_oi_alns_v2(_base().parents,FAST,cfg,ranker=ranker,**kwargs)
    assert learned.best_metrics==baseline.best_metrics
    assert learned.stats.proposal_trajectory==baseline.stats.proposal_trajectory
    assert learned.stats.raw_attempts==baseline.stats.raw_attempts==192
    assert learned.stats.attempted_by_family["ATOMIC"]==144
    assert learned.stats.attempted_by_family["LNS_REPAIRED"]==48
    assert learned.stats.kdp_count==baseline.stats.kdp_count<=8
    assert learned.stats.nref==baseline.stats.nref<=4
    assert learned.final_certification.certified and ranker.inference_seconds>0



def test_trajectory_transaction_keeps_partial_causal_capture(tmp_path):
    from test_search_sa import _base
    from types import SimpleNamespace
    solution=_base();entry={'instance_id':'fixture','split':'MLP_TRAIN','relative_path':'fixture.xlsx','tier':'SMALL'}
    context={'solution':solution,'directions':((),(),(0,0),(0,0)),'iteration':0,'elapsed':7.,'metrics':SimpleNamespace(cmax=100.)}
    # Use the actual serializable metrics type; partial stages are not fabricated.
    from mrta_reference.model import OfficialMetrics
    import inspect
    fields=inspect.signature(OfficialMetrics).parameters
    context['metrics']=OfficialMetrics(**{key:(100. if key=='cmax' else 0.) for key in fields})
    db=d.connect(tmp_path/'dataset.sqlite')
    d.store_trajectory(db,entry,solution.parents,20261121,{'missing_state_stages':['EARLY']},{'MID':context,'LATE':context})
    assert db.execute('SELECT count(*) FROM trajectories').fetchone()[0]==1
    assert [r[0] for r in db.execute('SELECT stage FROM states ORDER BY stage')]==['LATE','MID']
    with pytest.raises(sqlite3.IntegrityError): d.store_trajectory(db,entry,solution.parents,20261121,{}, {})
    assert db.execute('SELECT count(*) FROM states').fetchone()[0]==2
    db.close()



def test_candidate_labels_enforce_direction_stage_and_null_target(monkeypatch):
    from test_search_sa import _base,_family_candidates,FAST
    from mrta_search.lns import atomic_complete_candidate
    from mrta_search.direction import ConstrainedDirectionResult,DirectionStatus
    from mrta_reference.model import ScheduleResult,ScheduleStatus,OfficialMetrics
    from types import SimpleNamespace
    current=_base();directions=((),(),(0,0),(0,0))
    candidate=atomic_complete_candidate(current,directions,_family_candidates()[0],FAST)
    context={'solution':current,'directions':directions,'metrics':SimpleNamespace(cmax=100.)}
    monkeypatch.setattr(d,'CONFIG',FAST)
    absent=ConstrainedDirectionResult(DirectionStatus.NO_LEGAL_FIRST_ORIENTATION,((),(),(),()),None)
    monkeypatch.setattr(d,'optimize_directions_with_initial_feasibility',lambda *a:absent)
    def forbidden(*a,**k): raise AssertionError('Direction-infeasible candidate invoked reference')
    monkeypatch.setattr(d,'FormalReferenceEvaluator',forbidden)
    label=d.label_candidate(candidate,context,0)
    assert label['status']=='DIRECTION_INFEASIBLE' and label['c4'] is None
    assert label['y_feasible']==0 and label['y_improvement'] is None and label['reference_seconds']==0
    directed=ConstrainedDirectionResult(DirectionStatus.FEASIBLE,directions,5.)
    monkeypatch.setattr(d,'optimize_directions_with_initial_feasibility',lambda *a:directed)
    monkeypatch.setattr(d,'FormalReferenceEvaluator',lambda *a:lambda *a,**k:ScheduleResult(ScheduleStatus.DEADLOCK))
    label=d.label_candidate(candidate,context,0)
    assert label['status']=='DEADLOCK' and label['c4'] is not None
    assert label['y_feasible']==0 and label['y_improvement'] is None


def test_offline_heuristic_uses_exact_keys_and_no_label_inputs(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    import importlib.util
    from dataclasses import replace
    from types import SimpleNamespace
    from test_search_sa import _base,_family_candidates,FAST
    from mrta_ranker.features import extract_c2_features,extract_c4_features
    from mrta_search.lns import atomic_complete_candidate
    from mrta_search.direction import ConstrainedDirectionResult,DirectionStatus
    from mrta_reference.model import OfficialMetrics
    import inspect
    script=Path(__file__).resolve().parents[1]/'scripts/train_mlp_ranker.py'
    spec=importlib.util.spec_from_file_location('ranker_training_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    current=_base();directions=((),(),(0,0),(0,0));entry=next(e for e in d.representatives() if e['split']=='MLP_DEV')
    metrics=OfficialMetrics(**{k:(100. if k=='cmax' else 0.) for k in inspect.signature(OfficialMetrics).parameters})
    context={'solution':current,'directions':directions,'iteration':1,'elapsed':59.,'metrics':metrics}
    path=tmp_path/'dataset.sqlite';db=d.connect(path)
    d.store_trajectory(db,entry,current.parents,1,{}, {'LATE':context})
    state_id=entry['instance_id']+':1:LATE'
    candidate=atomic_complete_candidate(current,directions,_family_candidates()[0],FAST)
    with db:
        for i,travel in enumerate((10.,1.)):
            c=replace(candidate,projected_process_makespan=100.+(i+1)*1e-7)
            direct=ConstrainedDirectionResult(DirectionStatus.FEASIBLE,directions,travel)
            c2=extract_c2_features(current,directions,100.,c,FAST)
            c4=extract_c4_features(current,directions,100.,c,FAST,direction=direct,cheap_rank=i)
            d.set_meta(db,'C2_feature_names',list(c2));d.set_meta(db,'C4_feature_names',list(c4))
            db.execute('INSERT INTO candidates(state_id,candidate_id,family,payload,c2,c4,direction,status,y_feasible,y_improvement,reference_cmax) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (state_id,i,'STRUCTURAL',d.pack(c,current.parents),d.feature_blob(c2),d.feature_blob(c4),d.pack(direct),'FEASIBLE_CERTIFIED',1,.1,90.))
    db.close();connect=d.connect
    monkeypatch.setattr(d,'connect',lambda readonly=False:connect(path,readonly=readonly))
    rows,names=module.load_records('MLP_DEV')
    assert rows[0]['C2_features'][names['C2'].index('projected_process_after')]==rows[1]['C2_features'][names['C2'].index('projected_process_after')]
    assert rows[0]['projected']<rows[1]['projected']
    assert module.c4_select(rows,False,0,0)[0]['candidate_id']==0
    for r in rows: r['status']='DEADLOCK';r['reference_cmax']=None;r['y_improvement']=None
    assert module.c4_select(rows,False,0,0)[0]['candidate_id']==0

    pool=[{'Cs':100.,'status':'FEASIBLE_CERTIFIED','reference_cmax':c} for c in (90.,120.)]
    metric=module.capture_metrics(pool,[pool[1]])
    assert metric['capture']==0 and metric['normalized_regret']==pytest.approx(.3)
    assert metric['incumbent_fallback_regret']==pytest.approx(.1)
    metric=module.capture_metrics(pool,[{'status':'DEADLOCK','reference_cmax':None}])
    assert metric['capture']==0 and metric['normalized_regret'] is None
    assert module.capture_metrics([pool[1]],[pool[1]])['capture'] is None
    assert module.capture_metrics([dict(pool[0],reference_cmax=99.5)],[])['material']
    assert not module.capture_metrics([dict(pool[0],reference_cmax=99.50001)],[])['material']

    base={'capture_median':.1,'zero_capture':.6}
    assert module.qualifies_for_smoke(base,{'capture_median':.2,'zero_capture':.5})
    assert not module.qualifies_for_smoke(base,{'capture_median':.199,'zero_capture':.5})
    assert not module.qualifies_for_smoke(base,{'capture_median':.2,'zero_capture':.501})
    assert not module.qualifies_for_smoke(base,{'capture_median':None,'zero_capture':.5})
