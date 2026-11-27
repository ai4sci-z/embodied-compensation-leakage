"""Render fixed test-scene teacher videos for all three physical sources."""
from pathlib import Path
import json
from course_tools import render_rollout_media as media
ROOT=Path(__file__).resolve().parents[1]
media.MEDIA=ROOT/'course_results/three-source-v4/media'
protocol=json.loads((ROOT/'course_results/three-source-v2/collection_protocol.json').read_text())
seed=protocol['splits']['test'][0]
items=[]
for source,defect,label in [('delay','D2','控制延迟'),('backlash','D4hi','机械回差'),('droop','D3b','重力下垂')]:
 for level in (1,2):
  base='three-source-backlash-v4' if source=='backlash' else 'three-source-v2'
  p=ROOT/'course_results'/base/'data/Stack/Panda'/f'{defect}-L{level}'/f'seed-{seed}'/'collection.json'
  row=json.loads(p.read_text())['defective']
  items.append(media.render(Path(row['raw_path']),name=f'{source}-L{level}-seed-{seed}',title=f'{label} · 档位 {level} · 教师示教',category='three_source_teacher_demonstration',success=row['success']))
(media.MEDIA/'catalog.json').write_text(json.dumps({'selection':'first held-out test scene in collection protocol; both levels; no outcome selection','seed_env':seed,'items':items},ensure_ascii=False,indent=2)+'\n')
