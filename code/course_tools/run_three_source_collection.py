#!/usr/bin/env python3
"""Collect paired demonstrations for the three-source compensation study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runtime_results/engineering/learning-20261025-v1/causal-data'
OUT = ROOT / 'course_results/three-source-v2'
CELLS = [('delay', 'D2', 1), ('backlash', 'D4hi', 1), ('droop', 'D3b', 1),
         ('delay', 'D2', 2), ('backlash', 'D4hi', 2), ('droop', 'D3b', 2)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare():
    import h5py
    plan = json.loads((OLD / 'defective-plan.json').read_text())
    train_ids = set(plan['split']['train_episode_ids'])
    train = sorted({int(r['seed_env']) for r in plan['pool'] if r['id'] in train_ids})
    validation = sorted({int(r['seed_env']) for r in plan['pool'] if r['id'] not in train_ids})
    excluded = set(train + validation)
    for name in ['closed-loop-v1', 'recovery-evaluation-v1', 'recovery-evaluation-v2', 'recovery-evaluation-v3']:
        p = ROOT / 'course_results' / name / 'protocol.json'
        if p.exists():
            obj = json.loads(p.read_text())
            for key in ['new_evaluation_seeds', 'new_scene_seeds', 'evaluation_seeds', 'scene_seeds']:
                vals = obj.get(key, [])
                excluded.update(x for x in vals if isinstance(x, int))
    rng = random.Random(20261118)
    def extend(values, n):
        while len(values) < n:
            v = rng.randrange(1, 2**31 - 1)
            if v not in excluded:
                values.append(v)
                excluded.add(v)
        return values
    extend(train, 48)
    extend(validation, 12)
    test = extend([], 24)
    with h5py.File(plan['pool'][0]['path'], 'r') as h:
        thresholds = json.loads(h.attrs['thresholds_json'])
    p = {
        'schema': 'portal_course.three_source_collection/1',
        'task': 'Stack', 'family': 'Panda', 'control_hz': 20, 'horizon': 400,
        'cells': [dict(source=s, defect=d, level=l) for s,d,l in CELLS],
        'splits': {'train': train, 'validation': validation, 'test': test},
        'ilc_k_max': 20, 'ilc_tolerance_m': 0.01,
        'thresholds': thresholds,
        'threshold_role': 'historical D2 regime labels only; source attribution uses recorded physics and c_true, not presumed cross-defect calibrated regime thresholds',
        'clean_role': 'paired same-reference normal teacher for every cell; do not count duplicate normal scenes as independent data',
        'failed_demonstrations': 'retain all attempted episodes and quality; no replacement by additional seeds selected for success',
        'source_script_sha256': digest(__file__),
        'created_at_unix': time.time(),
    }
    path = OUT / 'collection_protocol.json'
    publish(OUT / 'regime_thresholds.json', {'thresholds': thresholds,
            'role': p['threshold_role']})
    if path.exists():
        existing = json.loads(path.read_text())
        for k in ['cells','splits','task','ilc_k_max','ilc_tolerance_m']:
            if existing[k] != p[k]:
                raise ValueError(f'existing collection protocol differs: {k}')
        return existing
    publish(path, p)
    return p


def episode_summary(path):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        g = f['data/demo_0']
        c = np.asarray(g['c_true'])[:, :6]
        error = np.asarray(g['tracking_error'])
        pos = np.asarray(g['eef_pos_ach'])
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 20
        actions = np.asarray(g['actions'])
        reversal = np.sum(np.diff(np.sign(actions[:, :3]), axis=0) != 0, axis=0)
        return {'path': str(path), 'sha256': digest(path),
                't_success': int(g.attrs['t_success']),
                'ilc_status': str(g.attrs['ilc_status']),
                'clean_ilc_status': str(g.attrs['clean_ilc_status']),
                'ilc_iterations': int(g.attrs['ilc_iters']),
                'c_rms': float(np.sqrt(np.mean(c*c))),
                'tracking_rms': float(np.sqrt(np.mean(error*error))),
                'speed_quantiles_mps': np.quantile(speed, [.1,.5,.9,.99]).tolist(),
                'cartesian_command_sign_changes': reversal.tolist(),
                'recorded_wrapper_state': list(g['defect_state'].shape),
                'recorded_gravity': 'tau_g' in g or 'surrogate_context' in g,
                'frames': len(actions)}


def collect_one(job):
    import contextlib
    from course_tools.collect_three_source import main as collect_main
    source, defect, level, split, seed, protocol = job
    cell = OUT / 'data' / 'Stack' / 'Panda' / f'{defect}-L{level}' / f'seed-{seed}'
    raw = cell / 'raw' / 'Stack' / 'Panda' / defect / f'L{level}' / 'demo.hdf5'
    clean = raw.parents[2] / f'clean-{defect}' / f'L{level}' / 'demo.hdf5'
    summary = cell / 'job.json'
    state = {'source': source, 'defect': defect, 'level': level, 'split': split,
             'seed_env': seed, 'pid': os.getpid(), 'started_at_unix': time.time()}
    if summary.exists():
        saved = json.loads(summary.read_text())
        if saved['status'] == 'complete':
            for side in ['defective', 'clean']:
                if digest(saved[side]['raw_path']) != saved[side]['raw_sha256']:
                    raise ValueError('existing raw file changed')
            return saved
        if saved['status'] == 'error':
            return saved
    publish(cell / 'progress.json', dict(state, status='collecting'))
    try:
        cell.mkdir(parents=True, exist_ok=True)
        with (cell / 'collection.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            collect_main(['--task','Stack','--family','Panda','--defect',defect,
                '--level',str(level),'--seed-env',str(seed),'--split',split,
                '--threshold-report',str(OUT/'regime_thresholds.json'),
                '--tol',str(protocol['ilc_tolerance_m']),'--k-max',str(protocol['ilc_k_max']),
                '--output-root',str(OUT/'data')])
        records = json.loads((cell/'collection.json').read_text())
        result = dict(state, status='complete', defective=records['defective'],
                      clean=records['clean'], completed_at_unix=time.time())
    except Exception as exc:
        result = dict(state, status='error', error=repr(exc), traceback=traceback.format_exc(),
                      completed_at_unix=time.time())
    publish(summary, result)
    publish(cell / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    protocol = prepare()
    if args.prepare_only:
        print(OUT / 'collection_protocol.json', flush=True)
        return
    jobs = [(s,d,l,split,seed,protocol) for split,seeds in protocol['splits'].items()
            for seed in seeds for s,d,l in CELLS]
    started = time.time()
    completed, errors, results = 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(collect_one, job) for job in jobs]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += r['status'] == 'complete'
            errors += r['status'] == 'error'
            status = {'planned_pairs': len(jobs), 'completed_pairs': completed, 'errors': errors,
                      'elapsed_seconds': time.time()-started, 'updated_at_unix': time.time(),
                      'latest': {k:r[k] for k in ['source','level','split','seed_env','status']}}
            publish(OUT / 'collection_progress.json', status)
            print(json.dumps(status), flush=True)
    publish(OUT / 'collection_summary.json', {'status':'complete' if not errors else 'completed_with_errors',
            'completed_pairs':completed, 'errors':errors, 'pairs':results})


if __name__ == '__main__':
    main()
