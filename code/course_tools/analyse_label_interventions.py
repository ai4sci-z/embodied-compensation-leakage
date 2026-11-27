#!/usr/bin/env python3
"""Paired label interventions from saved common-input predictions; no new inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from course_tools.evaluate_three_source import field_metrics, adjusted_metrics, direction_controls, smooth_reference
COMPONENTS={'translation':(slice(0,3),.05,2.5e-15),'rotation':(slice(3,6),.5,2.5e-13)}
BOOTSTRAP_SAMPLES=100000
BOOTSTRAP_SEED=20261118

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def interval(v,bootstrap_samples=BOOTSTRAP_SAMPLES):
 x=np.asarray([z for z in v if z is not None],float)
 if not len(x):return {'defined':0,'mean':None,'median':None,'ci95_scene_mean':None}
 rng=np.random.default_rng(BOOTSTRAP_SEED)
 bs=x[rng.integers(0,len(x),(bootstrap_samples,len(x)))].mean(1)
 return {'defined':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
         'ci95_scene_mean':np.quantile(bs,[.025,.975]).tolist(),'min':float(x.min()),'max':float(x.max())}

def process(d,out,bootstrap_samples=BOOTSTRAP_SAMPLES):
 protocol=json.loads((d/'protocol.json').read_text());summary=json.loads((d/'summary.json').read_text())
 if not protocol.get('contrast_sha256') or summary.get('status')!='complete':return None
 if summary['anchors']!=48 or len(summary['reports'])!=48:raise ValueError('expected two levels x 24 scenes')
 rows=[];directions=[];bindings=[]
 for name in summary['reports']:
  rp=d/Path(name).name;r=json.loads(rp.read_text());hp=d/Path(r['path']).name
  if not r['intervention_matched_training_verified'] or sha(hp)!=r['sha256']:raise ValueError('unverified intervention evidence')
  bindings.append({'report':str(rp.relative_to(ROOT)),'sha256':sha(rp),'data':str(hp.relative_to(ROOT)),'data_sha256':r['sha256']})
  with h5py.File(hp) as h:
   ref=h['anchor_actions_clean'][:].astype(np.float64);raw=h['prediction_clipped'][:].astype(np.float64);alt=h['contrast_prediction_clipped'][:].astype(np.float64)
   delta=raw-alt
   if not np.allclose(delta,h['intervention_delta_clipped'][:],atol=1.2e-7,rtol=0):raise ValueError('delta mismatch')
   for source in h['fields']:
    values={s:h[f'fields/{s}/c_source'][:] for s in h['fields']}
    for comp,(sl,scale,floor) in COMPONENTS.items():
     c=values[source][:,sl]*scale;a=(raw-ref)[:,sl]*scale;b=(alt-ref)[:,sl]*scale;effect=delta[:,sl]*scale
     mr,ma,md=[field_metrics(x,c,energy_floor=floor) for x in (a,b,effect)]
     nuisance={f'other_{s}':v[:,sl]*scale for s,v in values.items() if s!=source}
     q=ref[:,sl]*scale
     nuisance.update(reference_gain=q,reference_smoothing=smooth_reference(q)-q,
       source_reference_difference=(h[f'fields/{source}/actions_normal_source'][:]-ref)[:,sl]*scale)
     partial=adjusted_metrics(effect,c,nuisance,energy_floor=floor)
     partial['nuisance_adjustment_note']='Projection after other source fields, reference gain, reference smoothing and source-specific normal-reference difference; descriptive sensitivity analysis.'
     row={'seed_env':r['seed_env'],'level':r['level'],'source':source.split('-L')[0],
       'component':comp,'label_intervention':r['intervention_label_mode'],'step':protocol['checkpoint_step'],
       'training_seed':protocol['config']['seed'],'raw_lambda':mr['lambda'],'removed_lambda':ma['lambda'],
       'effect_lambda':md['lambda'],'effect_phi':md['phi'],'effect_cosine':md['direction_cosine'],
       'effect_rms':md['rmse'],'compensation_rms':md['compensation_rms'],
       'raw_reference_rmse':mr['rmse'],'removed_reference_rmse':ma['rmse'],
       'partial_effect_lambda':partial['lambda_partial'],
       'partial_compensation_residual_share':partial.get('compensation_residual_share'),
       'partial_status':partial['status']}
     if all(x is not None for x in (mr['lambda'],ma['lambda'],md['lambda'])):
      if abs(mr['lambda']-ma['lambda']-md['lambda'])>1e-7:raise ValueError('projection linearity identity failed')
     controls=[]
     for name,rot in direction_controls(c):
      rr=field_metrics(effect,rot,energy_floor=floor)
      controls.append({'name':name,'effect_lambda':rr['lambda'],'effect_cosine':rr['direction_cosine'],
        'cosine_to_true':field_metrics(c,rot,energy_floor=floor)['direction_cosine'],
        'identical_to_true':bool(np.allclose(c,rot,atol=1e-14,rtol=1e-10))})
     rows.append(row);directions.append({**row,'partial':partial,'rotated_controls':controls})
 metrics=['raw_lambda','removed_lambda','effect_lambda','effect_phi','effect_cosine','effect_rms','compensation_rms',
          'raw_reference_rmse','removed_reference_rmse','partial_effect_lambda','partial_compensation_residual_share']
 aggregates=[]
 for level in (1,2):
  for comp in COMPONENTS:
   for source in ('delay','backlash','droop'):
    group=[r for r in rows if (r['level'],r['component'],r['source'])==(level,comp,source)]
    if len(group)!=24 or len({r['seed_env'] for r in group})!=24:raise ValueError('scene duplication or missingness')
    aggregates.append({'level':level,'component':comp,'source':source,'scenes':24,
      **{m:interval([r[m] for r in group],bootstrap_samples) for m in metrics}})
 dest=out/d.name;dest.mkdir(parents=True,exist_ok=True)
 result={'schema':'dl_course.label_intervention_analysis/1','protocol':protocol,
   'label_intervention':rows[0]['label_intervention'],'rows':rows,'aggregates':aggregates,
   'confidence_interval':f'Paired test-scene bootstrap, {bootstrap_samples} resamples, conditional on this training seed; does not quantify variation over training seeds.',
   'bootstrap':{'replicates':bootstrap_samples,'seed':BOOTSTRAP_SEED,'unit':'paired scene',
     'scene_order':'Original summary report order retained within each group.',
     'pairing':'Raw, removed and difference values retain the same scene indices; arithmetic means give each scene equal weight.'},
   'direction_controls':'Fixed norm-preserving coordinate rotations; descriptive direction checks, not physical controls or a permutation null.',
   'source_fields':'Nonorthogonal source directions are reported separately, never summed.',
   'bindings':bindings,'analyser_sha256':sha(Path(__file__))}
 (dest/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
 (dest/'direction-analysis.json').write_text(json.dumps(directions,ensure_ascii=False,indent=2,allow_nan=False))
 with (dest/'scene-results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return dest

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evaluations',type=Path,default=ROOT/'course_results/three-source-evaluations-v4/common-probes/evaluations');ap.add_argument('--output',type=Path,default=ROOT/'course_results/three-source-evaluations-v4/label-analysis')
 ap.add_argument('--protocol',action='append',help='Only process this protocol directory; repeat for multiple protocols.')
 ap.add_argument('--bootstrap-samples',type=int,default=BOOTSTRAP_SAMPLES)
 args=ap.parse_args()
 if args.bootstrap_samples<100:ap.error('--bootstrap-samples must be at least 100')
 done=[]
 for p in sorted(args.evaluations.glob('*/summary.json')):
  if args.protocol and p.parent.name not in args.protocol:continue
  d=process(p.parent,args.output,args.bootstrap_samples)
  if d:done.append(str(d))
 print(json.dumps({'processed':done},ensure_ascii=False))
if __name__=='__main__':main()
