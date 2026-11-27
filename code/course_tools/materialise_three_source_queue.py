#!/usr/bin/env python3
"""Render completed physical demonstrations while collection continues."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from course_tools.run_three_source_collection import OUT, digest, publish


def materialise_one(record, cell, level, split, seed, defect):
    from course_tools.collect_three_source import materialise_record
    target = cell / 'observations' / f"{record['source']}.hdf5"
    done = cell / 'observations' / f"{record['source']}-result.json"
    if done.exists():
        return json.loads(done.read_text())
    try:
        row = materialise_record(record, cell, SimpleNamespace(level=level))
        result = dict(row, status='complete', raw_path=record['raw_path'],
            source=record['source'], level=level, split=split, seed_env=seed,
            defect=defect if record['source']!='clean' else 'clean',
            demo='demo_0', task='Stack', family='Panda',
            success=record['success'], ilc_status=record['ilc_status'],
            pair_id=record['pair_id'], mechanism_features=record['mechanism_features'])
    except Exception as exc:
        result = dict(status='error', raw_path=record['raw_path'], path=str(target),
                      error=repr(exc), traceback=traceback.format_exc())
    publish(done, result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    lock=(OUT/'materialise.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((OUT/'collection_protocol.json').read_text())
    expected=sum(map(len,protocol['splits'].values())) * (len(protocol['cells'])+1)
    jobs, outcomes, active = set(), {}, {}
    start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        while True:
            records=[]
            for p in sorted((OUT/'data').glob('*/*/*/*/job.json')):
                job=json.loads(p.read_text())
                if job['status']!='complete':
                    continue
                sides=['defective']
                if job['defect']=='D2' and job['level']==1:
                    sides.append('clean')
                for side in sides:
                    record=job[side]
                    key=record['raw_path']
                    if key in jobs:
                        continue
                    jobs.add(key)
                    result_file=p.parent/'observations'/f"{record['source']}-result.json"
                    if result_file.exists():
                        outcomes[key]=json.loads(result_file.read_text())
                    else:
                        records.append((key,record,p.parent,job))
            for key,record,cell,job in records:
                future=pool.submit(materialise_one,record,cell,job['level'],job['split'],job['seed_env'],job['defect'])
                active[future]=key
            if active:
                ready,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in ready:
                    key=active.pop(future)
                    try:
                        outcomes[key]=future.result()
                    except Exception as exc:
                        outcomes[key]={'status':'error','error':repr(exc),'raw_path':key}
            else:
                time.sleep(5)
            successes=[r for r in outcomes.values() if r['status']=='complete']
            errors=[r for r in outcomes.values() if r['status']=='error']
            progress={'expected_episodes':expected,'rendered':len(successes),'errors':len(errors),
                      'in_flight':len(active),'collection_seen':len(jobs),
                      'elapsed_seconds':time.time()-start,'updated_at_unix':time.time()}
            publish(OUT/'materialise_progress.json',progress)
            if not active and ((OUT/'collection_summary.json').exists() or len(outcomes)==expected):
                manifest={'schema':'dl_course.three_source_episodes/1','task':'Stack',
                    'collection_protocol_sha256':digest(OUT/'collection_protocol.json'),
                    'episodes':successes,'errors':errors,
                    'normal_source_selection':'D2-L1 paired normal reference, once per scene',
                    'all_attempts_retained':True}
                publish(OUT/'episodes.json',manifest)
                publish(OUT/'materialise_summary.json',dict(progress,status='complete' if not errors and len(successes)==expected else 'incomplete',errors_detail=errors))
                print(json.dumps(progress),flush=True)
                return


if __name__=='__main__':
    main()
