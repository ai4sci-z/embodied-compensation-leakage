"""Summarise observed demonstrations, retaining every collected scene."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course_results/three-source-v4/analysis'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for root in (ROOT/'course_results/three-source-v2',ROOT/'course_results/three-source-backlash-v4'):
 for p in sorted(root.glob('data/*/*/*/*/collection.json')):
  c=json.loads(p.read_text()); proto=json.loads((p.parent/'protocol.json').read_text())
  r=c['defective']; source=r['source']
  if source=='backlash' and root.name!='three-source-backlash-v4':continue
  f=r['mechanism_features']
  rows.append({'source':source,'level':proto['level'],'split':proto['split'],'scene':str(proto['seed_env']),
   'success':r['success'],'ilc_status':r['ilc_status'],'position_rms_m':r['position_rms_m'],
   'compensation_rms':f['compensation_rms_action_units'],
   'active_joint_reversals':f['active_joint_reversals'],
   'disengaged_fraction':f['active_disengaged_frame_fraction'],
   'max_penetration_rad':f.get('max_penetration_rad',float('nan')),
   'source_record':str(p),'source_record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
df=pd.DataFrame(rows);assert len(df)==504,len(df)
df.to_csv(OUT/'demonstrations.csv',index=False)
summary=[]
for (source,level),g in df.groupby(['source','level']):
 summary.append({'source':source,'level':int(level),'scenes':len(g),'successes':int(g.success.sum()),
 'converged':int((g.ilc_status=='converged').sum()),'compensation_median':float(g.compensation_rms.median()),
 'compensation_min':float(g.compensation_rms.min()),'compensation_max':float(g.compensation_rms.max()),
 'position_rms_median':float(g.position_rms_m.median()),'reversals_median':float(g.active_joint_reversals.median()),
 'disengaged_median':float(g.disengaged_fraction.median())})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
