"""Build the fixed workbook-isolated candidate dataset; resumable SQLite rows."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mrta_ranker import dataset as d
_WORKER_SOURCE=None
_PARENT_CACHE={}

def initialize_worker(masks):
    global _WORKER_SOURCE
    index=(mp.current_process()._identity[0]-1)%len(masks)
    d.bind_to_core(masks[index]);_WORKER_SOURCE=d.provenance()


def trajectory_job(entry,seed):
    if entry['instance_id'] not in _PARENT_CACHE: _PARENT_CACHE[entry['instance_id']]=d.load_parents(entry)
    parents=_PARENT_CACHE[entry['instance_id']]
    record,contexts=d.run_trajectory(entry,seed,parents=parents,source=_WORKER_SOURCE)
    db=d.connect()
    try: d.store_trajectory(db,entry,parents,seed,record,contexts)
    finally: db.close()
    return entry,seed,record


def state_job(state_id,action):
    db=d.connect();state=db.execute('SELECT * FROM states WHERE state_id=?',(state_id,)).fetchone()
    if action=='pools':
        count=d.generate_state_pool(db,state);db.close();return state_id,count
    context=d.state_context(db,state);parents=context['solution'].parents;count=0
    rows=list(db.execute('SELECT * FROM candidates WHERE state_id=? AND status IS NULL ORDER BY candidate_id',(state_id,)))
    for row in rows:
        candidate=d.unpack(row['payload'],parents)
        label=d.label_candidate(candidate,context,row['candidate_id'])
        with db:
            if label['c4_names'] is not None:
                old=d.metadata(db,'C4_feature_names')
                if old is not None and old!=label['c4_names']: raise ValueError('C4 schema drift')
                if old is None: d.set_meta(db,'C4_feature_names',label['c4_names'])
            db.execute('UPDATE candidates SET status=?,reference_cmax=?,y_feasible=?,y_improvement=?,c4=?,direction=?,direction_seconds=?,reference_seconds=? WHERE state_id=? AND candidate_id=? AND status IS NULL',
                tuple(label[k] for k in ('status','reference_cmax','y_feasible','y_improvement','c4','direction','direction_seconds','reference_seconds'))+(state_id,row['candidate_id']))
        count+=1
        if label['status']=='NUMERIC_FAILURE': raise RuntimeError('Numeric/certifier failure persisted; dataset cannot be trained')
    db.close();return state_id,count


def build(action,workers=3):
    split=d.validate_split(d.read(d.SPLIT));entries=d.representatives(split)
    db=d.connect()
    old=d.metadata(db,'representatives')
    if old is not None and old!=entries: raise ValueError('Existing representatives differ from fixed static selection')
    with db:
        d.set_meta(db,'representatives',entries);d.set_meta(db,'solver_seeds',list(d.SEEDS))
    masks=d.physical_core_affinities()[:workers]
    if len(masks)!=workers: raise ValueError('Not enough distinct physical cores')
    if action=='trajectories':
        done={tuple(row) for row in db.execute('SELECT instance_id,seed FROM trajectories')}
        jobs=[(e,seed) for e in entries for seed in d.SEEDS if (e['instance_id'],seed) not in done]
    elif action=='pools': jobs=[(row[0],action) for row in db.execute('SELECT state_id FROM states WHERE pool_complete=0')]
    else: jobs=[(row[0],action) for row in db.execute('SELECT DISTINCT state_id FROM candidates WHERE status IS NULL')]
    if action!='trajectories' and db.execute('SELECT count(*) FROM trajectories').fetchone()[0]!=46: raise ValueError('Need all 46 trajectory outcomes before candidate labeling')
    if action!='trajectories' and db.execute("SELECT count(*) FROM trajectories WHERE json_extract(result,'$.numeric_failure')>0").fetchone()[0]: raise ValueError('Numeric failure in a source trajectory must be resolved before training data generation')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker,initargs=(masks,)) as pool:
        futures=[pool.submit(trajectory_job,*job) if action=='trajectories' else pool.submit(state_job,*job) for job in jobs]
        for index,future in enumerate(as_completed(futures),1):
            result=future.result()
            if action=='trajectories':
                entry,seed,record=result
                print(f"trajectory {len(done)+index}/46 {entry['split']} {entry['tier']} seed={seed} iterations={record['iterations']} missing={record['missing_state_stages']}",flush=True)
            else: print(f'{action} {index}/{len(jobs)} candidates={result[1]}',flush=True)
    summary=d.dataset_summary(db)
    with db: d.set_meta(db,'summary',summary)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    print(d.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('trajectories','pools','labels'));parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();build(args.action,args.workers)
