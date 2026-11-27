"""Render the protocol's first held-out scene for each completed policy target."""
from pathlib import Path
import argparse
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    os.environ.setdefault('LP_NUM_THREADS', '1')
    from course_tools import render_rollout_media as media
    base = ROOT / 'course_results/three-source-evaluations-v4'
    media.MEDIA = base / 'media'
    media.MEDIA.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / 'course_results/three-source-v2/collection_protocol.json').read_text())
    scene = protocol['splits']['test'][0]
    labels = {'normal': '正常目标', 'D2': '延迟目标', 'D4hi': '回差目标', 'D3b': '下垂目标'}
    catalog_path = media.MEDIA / 'catalog.json'
    items = json.loads(catalog_path.read_text())['items'] if catalog_path.exists() else []
    done = {r['job'] for r in items}
    while True:
        jobs = json.loads((base / 'jobs.json').read_text())
        for key, job in jobs.items():
            if job['kind'] != 'rollout' or job['status'] != 'complete' or job['target'][2] != scene or key in done:
                continue
            row = json.loads(Path(job['result']).read_text())
            title = f"{job['step']//1000}k 更新 · {labels[row['target']]} · 档位 {row['level']} · 策略自主操作"
            result = media.render(Path(row['trajectory']), name=key, title=title,
                category='three_source_learned_policy_autonomous_rollout', success=row['success'])
            result.update(job=key, model=job['run'], checkpoint_step=job['step'],
                checkpoint_sha256=job['checkpoint_sha256'], seed_env=scene,
                target=row['target'], level=row['level'])
            items.append(result)
            done.add(key)
            tmp = catalog_path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'selection': 'first test scene in collection protocol; every target; no success selection',
                'items': items}, ensure_ascii=False, indent=2) + '\n')
            tmp.replace(catalog_path)
        if not args.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
