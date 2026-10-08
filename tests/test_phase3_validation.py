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



def test_fixed_ranker_loss_decomposition_and_negative_product():
    pytest.importorskip('torch')
    import importlib.util
    spec=importlib.util.spec_from_file_location('ranker_diagnostic_test',Path(__file__).resolve().parents[1]/'scripts/train_mlp_ranker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def row(i,cmax,family='STRUCTURAL',status='FEASIBLE_CERTIFIED'):
        return {'state_id':'s','tier':'LARGE','stage':'MID','Cs':100.,'seed':3,'iteration':1,
            'candidate_id':i,'family':family,'status':status,'C4_features':[1.],
            'reference_cmax':cmax,'y_feasible':int(status=='FEASIBLE_CERTIFIED'),
            'y_improvement':None if cmax is None else (100.-cmax)/100.}
    best=row(0,90.);partial=row(1,95.,'TARGET_X');worse=row(2,120.);dead=row(3,None,status='DEADLOCK')
    pool=[best,partial,worse,dead]
    # C2 misses the best but captures half the available improvement.
    loss=module.loss_decomposition(pool,[partial,worse,dead],[worse,dead])
    assert loss['material'] and loss['C2_capture']==.5 and loss['C4_capture']==0
    assert loss['oracle_C4_capture_given_C2']==.5 and loss['conditional_C4_capture']==0
    assert loss['loss_location']=='C4_DROPPED_ALL'
    none=module.loss_decomposition(pool,[worse,dead],[dead])
    assert none['loss_location']=='NO_C2_CAPTURE' and none['conditional_C4_capture'] is None
    retained=module.loss_decomposition(pool,[best,partial,worse],[partial])
    assert retained['conditional_C4_capture']==.5
    groups=module.decomposition_groups([loss,none,retained])['ALL']
    assert (groups['material_states'],groups['no_C2_capture_states'],groups['C4_dropped_all_states'])==(3,1,1)
    assert groups['conditional_C4_capture_median']==.25
    # Zero opportunity does not create a recall denominator or attribution.
    zero=module.loss_decomposition([worse],[worse],[worse])
    assert not zero['material'] and zero['loss_location']=='NO_POOL_OPPORTUNITY'
    for r,g,p in ((best,-.1,.9),(dead,-.2,.1)):
        r.update(C4_gain=g,C4_p=p,C4_score=g*p)
    negative=module.negative_product_diagnostics([best,dead],[dead],'C4')
    assert negative['inversions']==1 and negative['inversion_exists']
    dead.update(C4_gain=.2,C4_score=.02)
    assert not module.negative_product_diagnostics([best,dead],[dead],'C4')['inversion_exists']


@pytest.mark.parametrize('cmax,expected',[(995.01,0),(995.0,1),(994.99,1)])
def test_material_target_half_percent_boundary(cmax,expected):
    assert d.material_positive({'status':'FEASIBLE_CERTIFIED','Cs':1000.,'reference_cmax':cmax})==expected
    assert d.material_positive({'status':'DEADLOCK','Cs':1000.,'reference_cmax':None})==0


def test_pairwise_loss_moves_positive_above_negative():
    torch=pytest.importorskip('torch')
    from mrta_ranker.model import pairwise_loss,ranking_loss
    scores=torch.nn.Parameter(torch.tensor([-1.,1.]))
    opt=torch.optim.SGD([scores],lr=.5)
    for _ in range(8):
        opt.zero_grad();loss=pairwise_loss(scores[:1],scores[1:]);loss.backward();opt.step()
    assert scores[0]>scores[1]
    f=torch.zeros(2);m=torch.zeros(2);yf=torch.tensor([1.,0.]);ym=torch.tensor([1.,0.])
    total,a,b,c=ranking_loss(f,m,yf,ym,torch.tensor(3.),scores[:1],scores[1:])
    assert a==pytest.approx(.693147,rel=1e-5)
    assert b==pytest.approx(2*.693147,rel=1e-5)
    assert float(total.detach())==pytest.approx(float((a+b+c).detach()))


def _rank_fixture_pool():
    return [dict(state_id='state',candidate_id=i,family=('STRUCTURAL','TARGET_WHOLE','TARGET_Y','TARGET_X')[i%4],
                 status='FEASIBLE_CERTIFIED',Cs=1000.,reference_cmax=990. if i==39 else 1000.,
                 C2_score=float(i),C2_v1_score=float(i),C4_v1_score=float(i),
                 C4_features=[1.],projected=float(i),dp_travel=1.,split_delta=0.,seed=0,iteration=0,tier='LARGE',stage='LATE')
            for i in range(40)]


def test_c4_hard_subset_keeps_top8_and_material_excludes_direction_infeasible():
    pool=_rank_fixture_pool();pool[0].update(status='DIRECTION_INFEASIBLE',reference_cmax=None,C4_features=None)
    subset=d.c4_hard_subset(pool,pool[:8],pool[8:16],pool[16:24],lambda r:r['candidate_id'])
    ids={r['candidate_id'] for r in subset}
    assert set(range(1,24))<=ids  # Feasible parts of all three actual Top8 lists.
    assert set(range(24,32))<=ids and 39 in ids
    assert 0 not in ids and all(r['status']!='DIRECTION_INFEASIBLE' for r in subset)


def test_hard_pairs_are_state_local_bounded_and_state_equal():
    first=_rank_fixture_pool();second=[dict(r,state_id='other',reference_cmax=980. if r['candidate_id']>36 else 1000.) for r in _rank_fixture_pool()]
    rows=first+second
    pairs=d.hard_ranking_pairs(rows,'C2',lambda r:r['candidate_id'])
    from collections import Counter,defaultdict
    counts=Counter(p for p,n,w in pairs);weights=defaultdict(float)
    assert counts and max(counts.values())<=4
    for p,n,w in pairs:
        assert rows[p]['state_id']==rows[n]['state_id']
        assert d.material_positive(rows[p]) and not d.material_positive(rows[n])
        weights[rows[p]['state_id']]+=w
    assert list(weights.values())==pytest.approx([.5,.5])
    negatives={rows[n]['candidate_id'] for p,n,w in pairs if p==39}
    assert 0 in negatives and 38 in negatives  # heuristic-best and V1-best negatives.


def test_material_score_ignores_feasibility_and_signed_gain(tmp_path):
    torch=pytest.importorskip('torch');import numpy as np
    from mrta_ranker.model import CandidateMLP,predict_scores,save_model,load_model
    model=CandidateMLP(2,objective='material')
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.material.bias.fill_(-.05);model.feasibility.bias.fill_(9.)
    a=predict_scores(model,[[0.,0.]],np.zeros(2),np.ones(2))
    with torch.no_grad(): model.feasibility.bias.fill_(-9.)
    b=predict_scores(model,[[0.,0.]],np.zeros(2),np.ones(2))
    assert a[0]==b[0] and 0<=a[0]<=1
    assert not hasattr(model,'improvement')
    save_model(tmp_path/'material.pt',model,'C2',['a','b'],1,{})
    loaded,meta=load_model(tmp_path/'material.pt')
    assert meta['objective']=='material' and loaded.objective=='material'
    assert predict_scores(loaded,[[0.,0.]],np.zeros(2),np.ones(2))[0]==a[0]


def _training_module():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/train_mlp_ranker.py'
    spec=importlib.util.spec_from_file_location('rank_topk_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_c4_epoch_selection_uses_real_top8_and_strict_large_gate(monkeypatch):
    pytest.importorskip('torch');m=_training_module();pool=_rank_fixture_pool()
    original=m.c4_select;observed=[]
    def select(top8,*args):
        observed.append(len(top8));assert len(top8)<=8
        return original(top8,*args)
    monkeypatch.setattr(m,'c4_select',select)
    for r in pool:r['C4_score']=float(r['candidate_id'])
    m.evaluate_c4_top8(pool)
    assert observed==[8,8]
    base={'ALL':{'capture_median':0.,'zero_capture':.8},'LARGE':{'zero_capture':.5}}
    new={'ALL':{'capture_median':.1,'zero_capture':.7},'LARGE':{'zero_capture':.5}}
    assert m.rank_offline_gate(base,new)['passed']
    new['LARGE']['zero_capture']=.5001
    assert not m.rank_offline_gate(base,new)['passed']


def test_actual_material_inference_never_invokes_scheduler(monkeypatch):
    torch=pytest.importorskip('torch');import numpy as np
    from test_search_sa import _base,_family_candidates,FAST
    from mrta_ranker.ranking import LearnedRanker
    from mrta_ranker.model import CandidateMLP
    from mrta_ranker.features import extract_c2_features,extract_c4_features
    from mrta_search.lns import atomic_complete_candidate
    from mrta_search.pipeline import DirectionEvaluatedCandidate
    from mrta_search.direction import ConstrainedDirectionResult,DirectionStatus
    import mrta_reference.scheduler as scheduler
    def forbidden(*a,**kw):raise AssertionError('Inference invoked scheduler')
    monkeypatch.setattr(scheduler,'reference_schedule',forbidden)
    monkeypatch.setattr(d,'FormalReferenceEvaluator',forbidden)
    current=_base();directions=((),(),(0,0),(0,0))
    candidates=[atomic_complete_candidate(current,directions,c,FAST) for c in _family_candidates()]
    direct=ConstrainedDirectionResult(DirectionStatus.FEASIBLE,directions,1.)
    directed=[DirectionEvaluatedCandidate(c,i,direct) for i,c in enumerate(candidates)]
    names2=list(extract_c2_features(current,directions,100.,candidates[0],FAST))
    names4=list(extract_c4_features(current,directions,100.,candidates[0],FAST,direction=direct,cheap_rank=0))
    ranker=LearnedRanker.__new__(LearnedRanker);ranker.inference_seconds=0.
    ranker.models={s:(CandidateMLP(len(names),objective='material'),{'feature_names':names}) for s,names in (('C2',names2),('C4',names4))}
    ranker.scalers={s+suffix:(np.zeros(len(names)) if suffix=='_mean' else np.ones(len(names))) for s,names in (('C2',names2),('C4',names4)) for suffix in ('_mean','_std')}
    selected=ranker.select_c2(current,directions,100.,candidates,FAST,seed=0,iteration=0)
    assert len(selected)<=8
    assert len(ranker.select_c4(current,directions,100.,directed,FAST,seed=0,iteration=0))<=2
    assert ranker.inference_seconds>0


def test_existing_candidate_dataset_workbooks_still_disjoint():
    split=d.validate_split(d.read(d.SPLIT));db=d.connect(readonly=True)
    groups={role:{r[0] for r in db.execute('SELECT DISTINCT workbook FROM instances WHERE split=?',(role,))} for role in ('MLP_TRAIN','MLP_DEV')}
    assert groups['MLP_TRAIN']==set(split['MLP_TRAIN']) and groups['MLP_DEV']==set(split['MLP_DEV'])
    assert not groups['MLP_TRAIN']&groups['MLP_DEV']
    assert db.execute('SELECT count(*) FROM candidates').fetchone()[0]==14893
    db.close()
