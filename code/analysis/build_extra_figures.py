from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1]/'results'
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
plot_font = next((name for name in ('SimSun', 'Songti SC', 'Noto Serif CJK SC',
                                      'Source Han Serif SC', 'STSong')
                  if name in available_fonts), 'DejaVu Serif')
plt.rcParams.update({'font.family':plot_font,'font.size':15,'axes.unicode_minus':False,'mathtext.fontset':'stix','pdf.fonttype':42,'text.color':'black','axes.spines.top':False,'axes.spines.right':False})
F=R/'figures'
def save(f,n):
 f.savefig(F/(n+'.png'),dpi=180,bbox_inches='tight');f.savefig(F/(n+'.pdf'),bbox_inches='tight');plt.close(f)
S=pd.read_csv(R/'analysis-compensation/single_model.csv');S=S[S.variant=='clipped']
I=pd.read_csv(R/'analysis-compensation/label_interventions.csv');I=I[I.variant=='intervention_clipped']
colors=['#0072B2','#D55E00','#009E73'];sources=['delay','backlash','droop'];names=['延迟','回差','下垂']
for mode in ['raw','intervention']:
 fig,axs=plt.subplots(1,3,figsize=(11.5,3.7),layout='constrained')
 for j,(ax,src,name) in enumerate(zip(axs,sources,names)):
  if mode=='raw':d=S[(S.condition=='L1-balanced-raw')&(S.probe_level==1)&(S.component=='translation_command_m')&(S.source==src)]
  else:d=I[(I.condition=='L1-balanced-raw')&(I.probe_level==1)&(I.component=='translation_command_m')&(I.source==src)&(I.label_intervention=='remove-'+src)]
  vals=d.sort_values('seed').coefficient.to_numpy();ax.scatter(range(1,4),vals,s=65,c=colors[j]);ax.axhline(vals.mean(),c=colors[j],lw=1.5,ls='--');ax.axhline(0,c='black',lw=.7);ax.set_xticks([1,2,3],['重复 1','重复 2','重复 3']);ax.set_title(name);ax.set_ylabel('补偿方向系数' if mode=='raw' else '标签干预方向系数');ax.grid(axis='y',alpha=.12)
 save(fig,'slides-'+mode)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,probe in zip(axs,[1,2]):
 conds=['L1-balanced-raw','L1-balanced-remove-droop','L2-balanced-raw','L2-balanced-remove-all']
 for j,c in enumerate(conds):
  d=S[(S.condition==c)&(S.probe_level==probe)&(S.component=='translation_command_m')&(S.source=='droop')];v=d.coefficient.to_numpy();ax.scatter(j+np.linspace(-.1,.1,len(v)),v,c=colors[2],s=35);ax.plot([j-.22,j+.22],[v.mean()]*2,c='black',lw=2)
 ax.axhline(0,c='black',lw=.7);ax.set_xticks(range(4),['低强度\n保留','低强度\n去下垂','高强度\n保留','高强度\n全部去除'],fontsize=13);ax.set_title(f'{"低" if probe==1 else "高"}强度测试输入');ax.set_ylabel('下垂方向系数');ax.grid(axis='y',alpha=.15)
save(fig,'droop-strength')
J=json.loads((R/'evidence/teacher-crossover.json').read_text());d=pd.DataFrame(J['cells'])
fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained')
for ax,target,title in zip(axs,['clean','defective'],['正常目标身体','延迟目标身体']):
 for _,ds in d[d.target_body==target].groupby('seed'):
  v=[ds[ds.donor_feedforward==x].position_rms_m.iloc[0]*1000 for x in ['clean','defective']];ax.plot([0,1],v,'o-',c='#0072B2',alpha=.55,ms=4,lw=1)
 means=[d[(d.target_body==target)&(d.donor_feedforward==x)].position_rms_m.mean()*1000 for x in ['clean','defective']];ax.plot([0,1],means,'D-',c='black',ms=6,lw=2)
 ax.set_xticks([0,1],['正常前馈','延迟前馈']);ax.set_ylabel('位置跟踪 RMS（mm）');ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
save(fig,'teacher-matching')
J=json.loads((R/'evidence/label-aggregate.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained')
for row in J['per_training_seed']:
 d=row['equal_scene_means']
 for ax,ref in zip(axs,['clean','defective']):
  ax.plot([0,1],[d['raw_rmse_to_'+ref+'_teacher'],d['decompensated_rmse_to_'+ref+'_teacher']],'o-',color='#0072B2',alpha=.65)
for ax,t in zip(axs,['到正常教师的误差','到延迟教师的误差']):
 ax.set_title(t);ax.set_ylabel('六维动作 RMSE');ax.set_xticks([0,1],['保留补偿标签','去除补偿标签']);ax.grid(alpha=.15)
save(fig,'early-delay')
