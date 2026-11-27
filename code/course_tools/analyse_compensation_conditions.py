"""Describe measured teacher compensation at physically relevant recorded states."""
from pathlib import Path
import json,csv
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course_results/three-source-v4/analysis'
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
plot_font = next((name for name in ('SimSun', 'Songti SC', 'Noto Serif CJK SC',
                                      'Source Han Serif SC', 'STSong')
                  if name in available_fonts), 'DejaVu Serif')
plt.rcParams.update({'font.family':plot_font,'font.size':11,'pdf.fonttype':42,
 'svg.fonttype':'none','text.color':'black','axes.labelcolor':'black','font.weight':'normal',
 'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
rows=[]
manifest=json.loads((OUT.parent/'episodes.json').read_text())
for row in manifest['episodes']:
 if row['source']=='clean' or row['split']!='test':continue
 path=ROOT/Path(row['raw_path']).relative_to('/workspace/PORTAL-Trace')
 with h5py.File(path) as f:
  g=f['data/demo_0'];m=g['mechanism_features'];c=g['c_true'][:,:6]
  source=row['source'];level=row['level'];extra={}
  if source=='delay':
   x=m['reference_speed_m_s'][:];masks={'低速':x<=.02,'高速':x>.02}
   definition='reference speed <=20mm/s versus >20mm/s'
  elif source=='backlash':
   active=m['active_joint_indices'][:].astype(int)
   rev=m['joint_reversal'][:,active].any(axis=1)
   near=np.convolve(rev,np.ones(4),mode='full')[:len(c)]>0
   masks={'其余时刻':~near,'换向附近':near}
   definition='active motor-joint reversal and three following control steps versus other states'
   extra={'reversal_count':int(rev.sum()),'disengaged_fraction':float((~m['engaged'][:,active].astype(bool)).any(axis=1).mean())}
  else:
   x=np.linalg.norm(m['gravity_torque_nm'][:],axis=1);med=float(np.median(x))
   masks={'较小重力矩':x<=med,'较大重力矩':x>med};definition='gravity-torque norm below/above the within-episode median'
   extra={'gravity_median_nm':med}
  for group,mask in masks.items():
   for component,sl,scale in [('translation',slice(0,3),.05),('rotation',slice(3,6),.5)]:
    vals=c[mask,sl]*scale
    rows.append({'source':source,'level':level,'seed_env':row['seed_env'],'group':group,
     'component':component,'frames':int(mask.sum()),'rms':float(np.sqrt(np.mean(vals**2))) if mask.any() else None,
     'definition':definition,**extra})
(OUT/'compensation-conditions.json').write_text(json.dumps({'split':'test','scenes_per_level':24,'rows':rows,
 'interpretation':'Observed within-trajectory state strata; task stage and exposure are not independently randomized.'},ensure_ascii=False,indent=2))
groups={'delay':['低速','高速'],'backlash':['其余时刻','换向附近'],'droop':['较小重力矩','较大重力矩']}
titles={'delay':'延迟：参考运动速度','backlash':'回差：换向前后','droop':'下垂：构型重力矩'}
colors={1:'#0072B2',2:'#D55E00'}
fig,axes=plt.subplots(2,3,figsize=(11,6.1),layout='constrained')
summary=[]
for i,component in enumerate(['translation','rotation']):
 for j,source in enumerate(['delay','backlash','droop']):
  ax=axes[i,j]
  for level in (1,2):
   v=[r for r in rows if r['component']==component and r['source']==source and r['level']==level]
   seeds=sorted({r['seed_env'] for r in v});paired=[]
   for seed in seeds:
    vals=[next(r['rms'] for r in v if r['seed_env']==seed and r['group']==group) for group in groups[source]]
    if None in vals:continue
    paired.append(vals)
    ax.plot(np.array([0,1])+(level-1.5)*.07,vals,lw=.55,alpha=.22,color=colors[level])
   a=np.asarray(paired);med=np.median(a,axis=0)
   ax.plot(np.array([0,1])+(level-1.5)*.07,med,lw=1.8,color=colors[level],marker=['o','s'][level-1],label=f'档位 {level}')
   summary.append({'source':source,'component':component,'level':level,'paired_scenes':len(a),'groups':groups[source],'medians':med.tolist()})
  ax.set_yscale('log');ax.set_xticks([0,1],groups[source]);ax.set_xlim(-.2,1.2);ax.grid(axis='y',alpha=.12)
  if i==0:ax.set_title(titles[source]);ax.legend(frameon=False,fontsize=9)
  if j==0:ax.set_ylabel(['平移控制量补偿 RMS（m）','旋转控制量补偿 RMS（rad）'][i])
for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'compensation-conditions.{ext}',dpi=180)
plt.close(fig)
(OUT/'condition-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False))
