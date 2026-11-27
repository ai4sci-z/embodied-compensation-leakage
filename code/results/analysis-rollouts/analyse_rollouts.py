"""Aggregate saved rollout summaries/states without re-running policy inference."""
from pathlib import Path
import json, csv, collections
import numpy as np
import h5py

import os
ROOT=Path(os.environ.get('PORTAL_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
RESULT=ROOT/'course_results/formal-downstream-v1'
OUT=RESULT/'analysis-rollouts'
OUT.mkdir(exist_ok=True)
jobs=json.loads((RESULT/'jobs.json').read_text())
mapping={j['checkpoint_sha256']:j['run'] for j in jobs.values() if j['kind']=='rollout'}

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rows=[]
for p in sorted((RESULT/'rollouts/summaries').glob('*.json')):
    s=json.loads(p.read_text());run=mapping[s['checkpoint_sha256']]
    trajectory=ROOT/Path(s['trajectory']).relative_to('/workspace/PORTAL-Trace')
    r={k:s[k] for k in ['checkpoint_sha256','target','level','seed_env','training_seed','success','any_success','minimum_eef_cube_a_distance_m','cube_a_max_lift_m','peak_control_boundary_force_n','continuous_action_saturation_fraction','terminal_cube_xy_distance_m','wall_seconds']}
    r.update(run=run,condition=run.split('-seed-')[0],summary=p.name,trajectory=trajectory.name)
    with h5py.File(trajectory) as h:
        e=np.asarray(h['eef_position_pre_action_plus_terminal']); a=np.asarray(h['actions']);o=np.asarray(h['object_positions_pre_action_plus_terminal']);contact=np.asarray(h['task_contact_post_action']);succ=np.asarray(h['success_post_action'])
        r.update(contact_frames=int(contact.sum()),first_contact_s=float(np.flatnonzero(contact)[0]/20) if contact.any() else -1.,eef_final_x_m=float(e[-1,0]),eef_final_y_m=float(e[-1,1]),eef_final_z_m=float(e[-1,2]),eef_path_m=float(np.linalg.norm(np.diff(e,axis=0),axis=1).sum()),eef_displacement_m=float(np.linalg.norm(e[-1]-e[0])),eef_final_cube_a_distance_m=float(np.linalg.norm(e[-1]-o[-1,0])),action_translation_rms=float(np.sqrt(np.mean(a[:,:3]**2))),success_frames=int(succ.sum()),simulator_completed=True)
    rows.append(r)
write_csv('episodes.csv',rows)
metrics=['minimum_eef_cube_a_distance_m','cube_a_max_lift_m','peak_control_boundary_force_n','contact_frames','eef_path_m','eef_final_z_m','eef_final_cube_a_distance_m','action_translation_rms']
groups=collections.defaultdict(list)
for r in rows:groups[(r['condition'],r['target'],r['level'])].append(r)
out=[]
for (c,t,l),rs in sorted(groups.items()):
    v=dict(condition=c,target=t,level=l,n=len(rs),success=sum(r['success'] for r in rs),any_success=sum(r['any_success'] for r in rs),contact_episodes=sum(r['contact_frames']>0 for r in rs),lift_over_2cm=sum(r['cube_a_max_lift_m']>.02 for r in rs),lift_over_5cm=sum(r['cube_a_max_lift_m']>.05 for r in rs))
    for m in metrics:
        x=np.array([r[m] for r in rs]);v[m+'_mean']=float(x.mean());v[m+'_median']=float(np.median(x));v[m+'_std']=float(x.std(ddof=1));v[m+'_min']=float(x.min());v[m+'_max']=float(x.max())
    out.append(v)
write_csv('groups.csv',out)

idx={(r['condition'],r['training_seed'],r['target'],r['level'],r['seed_env']):r for r in rows}
pairs=[]
for r in rows:
    if '-remove-' not in r['condition']:continue
    base=r['condition'].split('-remove-')[0]+'-raw'
    ref=idx[(base,r['training_seed'],r['target'],r['level'],r['seed_env'])]
    v=dict(condition=r['condition'],training_seed=r['training_seed'],target=r['target'],level=r['level'],seed_env=r['seed_env'],raw_summary=ref['summary'],removed_summary=r['summary'],raw_trajectory=ref['trajectory'],removed_trajectory=r['trajectory'])
    for m in metrics:v[m+'_removed_minus_raw']=r[m]-ref[m]
    v.update(raw_success=ref['success'],removed_success=r['success'],raw_contact_frames=ref['contact_frames'],removed_contact_frames=r['contact_frames'])
    with h5py.File(RESULT/'rollouts/trajectories'/ref['trajectory']) as h1,h5py.File(RESULT/'rollouts/trajectories'/r['trajectory']) as h2:
        e1=np.asarray(h1['eef_position_pre_action_plus_terminal']);e2=np.asarray(h2['eef_position_pre_action_plus_terminal']);diff=e2-e1
        v.update(mean_paired_eef_separation_m=float(np.linalg.norm(diff,axis=1).mean()),terminal_paired_eef_separation_m=float(np.linalg.norm(diff[-1])),terminal_delta_x_m=float(diff[-1,0]),terminal_delta_y_m=float(diff[-1,1]),terminal_delta_z_m=float(diff[-1,2]))
    pairs.append(v)
write_csv('paired_episodes.csv',pairs)
pg=collections.defaultdict(list)
for r in pairs:pg[(r['condition'],r['target'],r['level'])].append(r)
pout=[]
for (c,t,l),rs in sorted(pg.items()):
    v=dict(condition=c,target=t,level=l,n=len(rs),raw_success=sum(r['raw_success'] for r in rs),removed_success=sum(r['removed_success'] for r in rs))
    ms=[k for k in rs[0] if k.endswith('_removed_minus_raw') or k in ['mean_paired_eef_separation_m','terminal_paired_eef_separation_m','terminal_delta_x_m','terminal_delta_y_m','terminal_delta_z_m']]
    for m in ms:
        vals=np.array([r[m] for r in rs]);v[m+'_mean']=float(vals.mean());v[m+'_median']=float(np.median(vals));v[m+'_positive_count']=int((vals>0).sum())
        seedvals=[np.mean([r[m] for r in rs if r['training_seed']==s]) for s in sorted({r['training_seed'] for r in rs})]
        for s,z in zip([1801,1802,1803],seedvals):v[m+'_seed'+str(s)]=float(z)
    pout.append(v)
write_csv('paired_groups.csv',pout)
selected=[]
for src,target in [('delay','D2'),('backlash','D4hi'),('droop','D3b')]:
    cand=[r for r in pairs if r['condition']=='L1-balanced-remove-'+src and r['target']==target and r['level']==1]
    cand.sort(key=lambda r:r['mean_paired_eef_separation_m']);p=cand[len(cand)//2]
    selected.append(dict(selection='median_paired_trajectory_separation',source=src,**p))
successes=[r for r in rows if r['success']]
selection={'paired_cases':selected,'successful_cases':successes,'total':len(rows),'success':len(successes),'any_success':sum(r['any_success'] for r in rows),'contact_episodes':sum(r['contact_frames']>0 for r in rows),'lift_over_2cm':sum(r['cube_a_max_lift_m']>.02 for r in rows),'lift_over_5cm':sum(r['cube_a_max_lift_m']>.05 for r in rows)}
(OUT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in selection.items() if k not in ['paired_cases','successful_cases']}))
