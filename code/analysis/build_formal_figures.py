from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[1]/'results'
F=R/'figures';F.mkdir(exist_ok=True)
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
plot_font = next((name for name in ('SimSun', 'Songti SC', 'Noto Serif CJK SC',
                                      'Source Han Serif SC', 'STSong')
                  if name in available_fonts), 'DejaVu Serif')
plt.rcParams.update({'font.family':plot_font,'font.size':11,'axes.unicode_minus':False,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
S=pd.read_csv(R/'analysis-compensation/single_model.csv'); S=S[S.variant=='clipped']
I=pd.read_csv(R/'analysis-compensation/label_interventions.csv');I=I[I.variant=='intervention_clipped']
G=pd.read_csv(R/'analysis-rollouts/groups.csv');P=pd.read_csv(R/'analysis-rollouts/paired_groups.csv')
colors=['#0072B2','#D55E00','#009E73'];src=['delay','backlash','droop'];cn=['延迟','回差','下垂']
comp=['translation_command_m','rotation_command_rad']
conditions=['L1-balanced-raw','L1-balanced-remove-delay','L1-balanced-remove-backlash','L1-balanced-remove-droop','L1-balanced-remove-all','clean-only']
labels=['保留全部','去延迟','去回差','去下垂','全部去除','正常示教']
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.12)
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
 plt.close(fig)
# Direct scene-level demonstration truth; paired points are not means only.
D=pd.read_csv(R/'demonstration-components.csv')
print('truth cols',D.columns.tolist())
for c in comp:
 fig,axs=plt.subplots(2,3,figsize=(11.8,6.0),layout='constrained')
 for l in [1,2]:
  for k,source in enumerate(src):
   ax=axs[l-1,k]
   for j,cond in enumerate(conditions):
    d=S[(S.condition==cond)&(S.probe_level==l)&(S.component==c)&(S.source==source)].sort_values('seed')
    a=d.coefficient.to_numpy()
    ax.scatter(j+np.linspace(-.12,.12,len(a)),a,s=25,color=colors[k],zorder=3)
    if len(a): ax.plot([j-.22,j+.22],[a.mean()]*2,c='black',lw=2)
   ax.axhline(0,c='#999999',lw=.7);ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right',fontsize=9)
   ax.set_title(f'{cn[k]}方向 · {"低" if l==1 else "高"}强度探针',fontsize=12)
   ax.set_ylabel(r'投影系数 $\Lambda$'); ax.grid(axis='y',alpha=.16)
 save(fig,'inheritance-'+('translation' if c==comp[0] else 'rotation'))
fig,axs=plt.subplots(2,2,figsize=(10.2,6.3),layout='constrained')
for li,l in enumerate([1,2]):
 for ci,c in enumerate(comp):
  vals=np.zeros((4,3))
  for ii,mode in enumerate(['remove-delay','remove-backlash','remove-droop','remove-all']):
   for j,source in enumerate(src):
    d=I[(I.condition=='L1-balanced-raw')&(I.label_intervention==mode)&(I.probe_level==l)&(I.component==c)&(I.source==source)]
    vals[ii,j]=d.coefficient.mean()
  ax=axs[li,ci];v=max(abs(vals).max(),.05);im=ax.imshow(vals,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto')
  for ii in range(4):
   for j in range(3):ax.text(j,ii,f'{vals[ii,j]:.3f}',ha='center',va='center',fontsize=12,color='white' if abs(vals[ii,j])>.68*v else 'black')
  ax.set_xticks(range(3),cn);ax.set_yticks(range(4),['去延迟','去回差','去下垂','全部去除']);ax.set_title(f'{"低" if l==1 else "高"}强度探针 · {"平移" if ci==0 else "旋转"}')
  fig.colorbar(im,ax=ax,shrink=.8,label='原策略 - 去补偿策略的方向系数')
save(fig,'intervention-matrix')
fig,axs=plt.subplots(1,3,figsize=(11.5,3.7),layout='constrained')
for j,source in enumerate(src):
 ax=axs[j]
 for seed in sorted(S.seed.unique()):
  a=[]
  for cond in ['L1-balanced-raw',f'L1-emphasize-{source}-raw']:
   d=S[(S.condition==cond)&(S.seed==seed)&(S.source==source)&(S.component==comp[0])&(S.probe_level==1)]
   a.append(float(d.iloc[0].coefficient))
  ax.plot([25,62.5],a,'o-',c=colors[j],alpha=.65,lw=1.3,ms=5)
 ax.axhline(0,c='#888888',lw=.7);ax.set_xticks([25,62.5],['25%','62.5%']);ax.set_xlabel('该来源的采样比例');ax.set_ylabel(r'平移投影系数 $\Lambda$');ax.set_title(cn[j]);ax.grid(alpha=.15)
save(fig,'mixture-effects')
# Target heatmap: contact and close approach rather than just success.
targets=[('normal',1),('D2',1),('D2',2),('D4hi',1),('D4hi',2),('D3b',1),('D3b',2)]
tlabels=['正常','延迟50 ms','延迟100 ms','回差0.1°','回差0.3°','下垂10%','下垂20%']
fig,axs=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for ci,(col,title,scale) in enumerate([('minimum_eef_cube_a_distance_m_median','距目标方块的最小距离中位数（mm）',1000),('contact_episodes','发生方块间接触的轨迹数（每格72条）',1)]):
 vals=np.array([[float(G[(G.condition==cond)&(G.target==t)&(G.level==l)].iloc[0][col])*scale for t,l in targets] for cond in conditions])
 ax=axs[ci];im=ax.imshow(vals,cmap='Blues',aspect='auto',vmin=0)
 for i in range(len(conditions)):
  for j in range(7):ax.text(j,i,f'{vals[i,j]:.0f}',ha='center',va='center',fontsize=9,color='white' if vals[i,j]>.65*vals.max() else 'black')
 ax.set_xticks(range(7),tlabels,rotation=42,ha='right',fontsize=9);ax.set_yticks(range(6),labels);ax.set_title(title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.8)
save(fig,'target-outcomes')
fig,axs=plt.subplots(1,3,figsize=(11.5,4),layout='constrained')
for j,source in enumerate(src):
 ax=axs[j];d=P[P.condition==f'L1-balanced-remove-{source}']
 vals=[]
 for ti,(t,l) in enumerate(targets):
  row=d[(d.target==t)&(d.level==l)].iloc[0]
  y=[float(row[f'mean_paired_eef_separation_m_seed{s}'])*1000 for s in [1801,1802,1803]]
  ax.scatter(ti+np.linspace(-.12,.12,3),y,c=colors[j],s=24);ax.plot([ti-.2,ti+.2],[np.mean(y)]*2,c='black',lw=1.6)
 ax.set_title(f'去{cn[j]}补偿');ax.set_xticks(range(7),tlabels,rotation=48,ha='right',fontsize=8);ax.set_ylabel('配对轨迹平均末端间距（mm）');ax.grid(axis='y',alpha=.15)
save(fig,'paired-consequences')
(F/'figure-notes.json').write_text(json.dumps({'inheritance':'每点一个训练种子，先将24场景的内积和能量求和再计算Lambda；黑线三种子均值。','intervention':'每格为3个训练种子的均值，误差改变为原策略减去补偿策略。','mixture':'同色线连接同种子两种采样比例，平移方向，低强度探针。','target':'每格3种子×24场景=72条，距离为全部72条中位数；接触为轨迹数。','paired':'每点为一个训练种子在24场景的平均配对末端距离；不是动作误差。'},ensure_ascii=False,indent=2))
print('figures done')

fig,axs=plt.subplots(2,3,figsize=(11.5,5.6),layout='constrained')
for ci,col in enumerate(['translation_rms','rotation_rms']):
 for j,source in enumerate(src):
  ax=axs[ci,j];d=D[D.source==source]
  wide=d.pivot(index='scene',columns='level',values=col)
  for _,row in wide.iterrows():ax.plot([1,2],[row[1],row[2]],color=colors[j],alpha=.2,lw=.6)
  ax.scatter(np.ones(len(wide)),wide[1],s=7,c=colors[j],alpha=.5);ax.scatter(np.full(len(wide),2),wide[2],s=7,c=colors[j],alpha=.5)
  ax.scatter([1,2],[wide[1].median(),wide[2].median()],marker='D',s=35,c='black',zorder=3)
  ax.set_yscale('log');ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}'));ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_xticks([1,2],[['50 ms','100 ms'],['0.1°','0.3°'],['10%','20%']][j]);ax.set_ylabel(('平移' if ci==0 else '旋转')+'补偿 RMS');ax.set_title(cn[j] if ci==0 else '');ax.grid(axis='y',alpha=.15)
save(fig,'truth-amplitude')
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def box(ax,x,y,w,h,text,fill='#EDF5FA'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.05',facecolor=fill,edgecolor='#0072B2',lw=1.1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arrow(ax,x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#0072B2'))
fig,ax=plt.subplots(figsize=(11.5,4.6));ax.set(xlim=(0,12),ylim=(0,4.5));ax.axis('off')
for y,t in [(3.1,'外部相机\n5 帧 RGB'),(1.85,'腕部相机\n5 帧 RGB'),(.6,'本体历史\n5 × 70')]:box(ax,.1,y,1.9,.85,t)
for y,t in [(3.1,'ResNet18\n45 个视觉 token'),(1.85,'ResNet18\n45 个视觉 token'),(.6,'状态 MLP\n5 个状态 token')]:box(ax,2.5,y,2.3,.85,t);arrow(ax,2,y+.425,2.5,y+.425)
box(ax,5.5,1.8,2.5,1.15,'加入任务嵌入\n96 × 512 观察特征')
for y in [3.525,2.275,1.025]:arrow(ax,4.8,y,5.5,2.375)
box(ax,8.6,1.8,3.15,1.15,'4 层 Transformer\n8 头注意力 · 8 个查询');arrow(ax,8,2.375,8.6,2.375)
box(ax,8.6,.25,3.15,.85,'8 × 7 动作块');arrow(ax,10.175,1.8,10.175,1.1)
ax.text(5.5,.55,'每步重新观察，执行预测块的第一步',fontsize=12,ha='center')
save(fig,'network')
fig,ax=plt.subplots(figsize=(11.5,3.4));ax.set(xlim=(0,12),ylim=(0,3.3));ax.axis('off')
for x,t in [(.15,'正常、延迟、回差、下垂\n同场景教师示教'),(4.1,'混合观察与动作标签\n共享策略训练'),(8.05,'正常及三类目标身体\n自主执行')]:box(ax,x,1.5,3.55,1.05,t)
arrow(ax,3.7,2.025,4.1,2.025);arrow(ax,7.65,2.025,8.05,2.025)
for x,t in [(.15,'同状态教师差：补偿真值'),(4.1,'共同输入：投影与标签干预'),(8.05,'配对场景：轨迹、接触、任务')]:ax.text(x+1.775,.65,t,ha='center',va='center',fontsize=11)
save(fig,'workflow')
