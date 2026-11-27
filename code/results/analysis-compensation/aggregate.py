"""Aggregate saved common-input predictions; no model inference or training.

Usage: python aggregate.py
Scene-level dot products and energies are summed before forming lambda/phi.
Training seeds remain separate replicates throughout the primary CSV.
"""
from pathlib import Path
import json, csv, math, statistics
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'

def condition(cfg):
    w=cfg['source_weights']
    if w.get('clean')==1: return 'clean-only'
    special=[k for k,v in w.items() if v>.5]
    base='emphasize-'+special[0] if special else 'balanced'
    return 'L%d-%s-%s'%(cfg['level'],base,cfg['label_mode'])

def write_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def derived(rows):
    dot=sum(r['dot_sum'] for r in rows)
    ce=sum(r['compensation_energy_sum'] for r in rows)
    ee=sum(r['error_energy_sum'] for r in rows)
    n=sum(r['frames'] for r in rows); dim=rows[0]['dimensions']
    return dict(scenes=len(rows),frames=n,dot=dot,compensation_energy=ce,error_energy=ee,
        coefficient=dot/ce if ce else None,energy_share=dot*dot/(ce*ee) if ce and ee else None,
        cosine=dot/math.sqrt(ce*ee) if ce and ee else None,
        compensation_rms=math.sqrt(ce/(n*dim)),error_rms=math.sqrt(ee/(n*dim)),
        projected_rms=abs(dot)/math.sqrt(ce*n*dim) if ce else None)

def main():
    single=[]; pairs=[]; phase=[]; scenes=[]; models=[]; truth=[]; geometries=[]; gaps=[]
    processed_truth=False
    for path in sorted(RAW.glob('*/summary.json')):
        p=json.loads((path.parent/'protocol.json').read_text())
        cfg=p['config']; cond=condition(cfg); seed=cfg['seed']
        reports=[json.loads(q.read_text()) for q in sorted(path.parent.glob('backlash-L*.json'))]
        if len(reports)!=48:raise ValueError((path,len(reports)))
        if not p['contrast_sha256']:
            models.append(dict(condition=cond,seed=seed,checkpoint=p['checkpoint_sha256'],
                protocol=p['protocol_sha256'] if 'protocol_sha256' in p else path.parent.name,
                train_level=cfg['level'],label_mode=cfg['label_mode'],source_weights=cfg['source_weights'],
                saturation_fraction=sum(r['saturation_fraction']*r['frames'] for r in reports)/sum(r['frames'] for r in reports)))
        for variant in ('unclipped','clipped','intervention_unclipped','intervention_clipped'):
            if variant not in reports[0]:continue
            intervention=variant.startswith('intervention')
            if not intervention and p['contrast_sha256']:continue
            alt=reports[0].get('intervention_label_mode','') if intervention else ''
            for level in (1,2):
                rr=[r for r in reports if r['level']==level]
                for component in rr[0][variant]:
                    for source in rr[0][variant][component]:
                        meta=dict(condition=cond,seed=seed,label_intervention=alt,probe_level=level,component=component,
                                  source=source.split('-L')[0],variant=variant,checkpoint=p['checkpoint_sha256'],
                                  contrast_checkpoint=p['contrast_sha256'] or '')
                        group=[r[variant][component][source]['all'] for r in rr]
                        result={**meta,**derived(group)}
                        (pairs if intervention else single).append(result)
                        for r,g in zip(rr,group):
                            scenes.append({**meta,'scene':r['seed_env'],**derived([g])})
                        for ph in sorted(set.intersection(*[set(r[variant][component][source]) for r in rr])):
                            if ph=='all':continue
                            vals=[r[variant][component][source][ph] for r in rr]
                            vals=[v for v in vals if v['frames']]
                            if vals:phase.append({**meta,'phase':ph,**derived(vals)})
        if not processed_truth:
            for level in (1,2):
                rr=[r for r in reports if r['level']==level]
                for comp in rr[0]['unclipped']:
                    for src in rr[0]['unclipped'][comp]:
                        vals=[r['unclipped'][comp][src]['all'] for r in rr]
                        ag=derived(vals)
                        truth.append(dict(probe_level=level,source=src.split('-L')[0],component=comp,
                           frames=ag['frames'],compensation_energy=ag['compensation_energy'],compensation_rms=ag['compensation_rms']))
                for r in rr:
                    for comp,g in r['field_geometry'].items():
                        for i,a in enumerate(g['labels']):
                            for j,b in enumerate(g['labels']):
                                geometries.append(dict(level=level,scene=r['seed_env'],component=comp,source_a=a,source_b=b,cosine=g['normalized_gram'][i][j]))
                    for source,gap in r['normal_source_minus_fixed_anchor_reference'].items():
                        gaps.append(dict(level=level,scene=r['seed_env'],source=source,**gap))
            processed_truth=True
    if len(models)!=33:raise ValueError(('model_count',len(models)))
    write_csv('single_model.csv',single);write_csv('label_interventions.csv',pairs)
    write_csv('scene_readings.csv',scenes);write_csv('phase_readings.csv',phase)
    write_csv('truth_rms.csv',truth);write_csv('direction_geometry.csv',geometries);write_csv('reference_gaps.csv',gaps)
    (ROOT/'models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2))
    grouped=defaultdict(list)
    for row in single+pairs:
        key=tuple((k,row[k]) for k in ('condition','label_intervention','probe_level','component','source','variant'))
        grouped[key].append(row)
    summaries=[]
    for key,rr in sorted(grouped.items()):
        row=dict(key);row['n_seeds']=len(rr)
        for metric in ('coefficient','energy_share','error_rms','projected_rms'):
            vs=[r[metric] for r in rr]
            row[metric+'_mean']=statistics.mean(vs);row[metric+'_sd']=statistics.stdev(vs) if len(vs)>1 else 0
            row[metric+'_min']=min(vs);row[metric+'_max']=max(vs)
        summaries.append(row)
    write_csv('condition_summary.csv',summaries)
    (ROOT/'aggregate.json').write_text(json.dumps(dict(models=models,truth=truth,single_model=single,label_interventions=pairs,
       condition_summary=summaries,method='Within training seed, sum scene dot products/energies, then compute projection. Across seeds report mean and sample SD. Probe trajectories are backlash-anchored, 24 scenes x 2 levels. No independent attribution is implied by projection. Translation command scaled by 0.05 m and rotation command by 0.5 rad.'),ensure_ascii=False,indent=2))
    print(json.dumps(dict(models=len(models),single_rows=len(single),pair_rows=len(pairs),scene_rows=len(scenes)),indent=2))
    for rr in summaries:
        if rr['probe_level']==1 and rr['component']=='translation_command_m' and rr['variant'] in ('unclipped','intervention_unclipped'):
            print(rr['condition'],rr['label_intervention'],rr['source'],'lambda %.5f +/- %.5f phi %.5f'%(rr['coefficient_mean'],rr['coefficient_sd'],rr['energy_share_mean']))

if __name__=='__main__':main()
