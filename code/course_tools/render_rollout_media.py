#!/usr/bin/env python3
"""Render actual saved MuJoCo states into full-resolution dual-view media."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "course_results/closed-loop-v1"
MEDIA = STUDY / "media"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def render(source, *, name, title, category, success, media_root=None,
           checkpoint_sha256=None, source_summary=None):
    output = (Path(media_root) if media_root else MEDIA) / name
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "media.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["source_sha256"] != digest(source):
            raise ValueError("Source trajectory changed after rendering")
        if old.get("render_revision") == 3:
            if old.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError("Saved media checkpoint binding changed")
            if (output / "dual-view.mp4").is_file() and digest(output / "dual-view.mp4") == old["video_sha256"]:
                for filename, expected in old.get("image_sha256", {}).items():
                    if digest(output / filename) != expected:
                        raise ValueError("Saved media image changed")
                return old
    with h5py.File(source, "r") as handle:
        if "states_pre_action_plus_terminal" in handle:
            states = np.asarray(handle["states_pre_action_plus_terminal"])
            xml = handle.attrs["model_xml"]
        else:
            demo = handle["data/demo_0"]
            states = np.asarray(demo["states"])
            xml = demo.attrs["model_file"]
    if isinstance(xml, bytes):
        xml = xml.decode()
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offheight = max(512, model.vis.global_.offheight)
    model.vis.global_.offwidth = max(512, model.vis.global_.offwidth)
    data = mujoco.MjData(model)
    if states.ndim != 2 or states.shape[1] != model.nq + model.nv or not np.isfinite(states).all():
        raise ValueError("Invalid saved physics states for this model")
    renderer = mujoco.Renderer(model, height=512, width=512)
    scene_option = mujoco.MjvOption()
    # Match robosuite's default camera rendering: visual meshes, no collision meshes.
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1
    font = ImageFont.truetype(str(ROOT / "course_assets/fonts/Simsun.ttc"), 23)
    small = ImageFont.truetype(str(ROOT / "course_assets/fonts/Simsun.ttc"), 21)
    times = ImageFont.truetype(str(ROOT / "course_assets/fonts/TimesNewRoman.ttf"), 24)
    temporary = output / "dual-view.partial.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1024x576", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-crf", "20",
           "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temporary)]
    selected = [0, 80, 160, 240, 320, len(states) - 1]
    sheet = Image.new("RGB", (1536, 566), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text((12, 6), title, font=font, fill="black")
    start = time.monotonic()
    with (output / "encoding.log").open("w") as error_log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=error_log)
        try:
            for index, state in enumerate(states):
                data.qpos[:] = state[:model.nq]
                data.qvel[:] = state[model.nq:]
                data.time = index / 20
                mujoco.mj_forward(model, data)
                frame = Image.new("RGB", (1024, 576), "white")
                draw = ImageDraw.Draw(frame)
                draw.text((12, 4), title, font=font, fill="black")
                draw.text((12, 34), "外部视角", font=small, fill="black")
                draw.text((524, 34), "腕部视角", font=small, fill="black")
                draw.text((865, 34), f"t = {index / 20:.2f} s", font=times, fill="black")
                for col, camera in enumerate(("agentview", "robot0_eye_in_hand")):
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    image = Image.fromarray(renderer.render().copy())
                    frame.paste(image, (col * 512, 64))
                    if index in selected:
                        k = selected.index(index)
                        sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (k * 256, 54 + col * 256))
                process.stdin.write(frame.tobytes())
                if index in selected:
                    frame.save(output / f"frame-{index:04d}.png")
                    k = selected.index(index)
                    sheet_draw.text((k * 256 + 12, 31), f"{index / 20:.2f} s", font=small, fill="black")
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError((output / "encoding.log").read_text())
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            renderer.close()
    temporary.replace(output / "dual-view.mp4")
    sheet.save(output / "keyframes.png")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(output / "dual-view.mp4")], text=True))
    stream = probe["streams"][0]
    if int(stream["nb_frames"]) != len(states) or (stream["width"], stream["height"]) != (1024, 576):
        raise ValueError("Encoded video frame count or dimensions differ")
    result = {"render_revision": 3, "source": str(source), "source_sha256": digest(source), "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "category": category, "title": title, "terminal_task_success": bool(success),
        "rendering": "MuJoCo forward rendering of recorded qpos/qvel; no new physics integration or policy inference",
        "native_camera_pixels": [512, 512], "video_pixels": [1024, 576], "fps": 20,
        "visual_geometry_groups": {"collision_group_0": False, "visual_group_1": True},
        "frames": len(states), "keyframe_indices": selected, "video": str(output / "dual-view.mp4"),
        "video_sha256": digest(output / "dual-view.mp4"), "probe": stream,
        "image_sha256": {p.name: digest(p) for p in sorted(output.glob("*.png"))},
        "checkpoint_sha256": checkpoint_sha256,
        "source_summary": str(source_summary) if source_summary else None,
        "source_summary_sha256": digest(source_summary) if source_summary else None,
        "renderer_sha256": digest(__file__),
        "fonts": {name: digest(ROOT / "course_assets/fonts" / name)
                  for name in ("Simsun.ttc", "TimesNewRoman.ttf")},
        "text_style": "regular SimSun Chinese; Times New Roman time labels; black text",
        "wall_seconds": time.monotonic() - start}
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"name": name, "frames": len(states), "seconds": result["wall_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--title")
    parser.add_argument("--category", default="learned_policy_autonomous_rollout")
    parser.add_argument("--success", action="store_true")
    args = parser.parse_args()
    if args.source:
        if not args.output or not args.name or not args.title:
            parser.error("single-trajectory rendering requires --output, --name and --title")
        if args.source_summary:
            summary = json.loads(args.source_summary.read_text())
            if (summary["trajectory_sha256"] != digest(args.source)
                    or summary["checkpoint_sha256"] != args.checkpoint_sha256
                    or bool(summary["success"]) != args.success):
                raise ValueError("Media source differs from the saved rollout summary")
        render(args.source, name=args.name, title=args.title, category=args.category,
               success=args.success, media_root=args.output,
               checkpoint_sha256=args.checkpoint_sha256, source_summary=args.source_summary)
        return
    protocol = json.loads((STUDY / "protocol.json").read_text())
    scenes = [("development", protocol["development_seeds"][0]),
              ("new_evaluation", protocol["new_evaluation_seeds"][0])]
    catalog, pending = [], []
    for model in protocol["models"]:
        label = {"raw": "原始标签", "decompensated": "去补偿标签", "clean_demonstrations": "BC-RNN 参考"}[model["label_mode"]]
        for split, seed in scenes:
            for deployment in protocol["deployments"]:
                name = f"{model['id']}-{deployment}-{split}-{seed}"
                path = STUDY / "summaries" / f"{name}.json"
                if not path.exists():
                    pending.append(name)
                    continue
                row = json.loads(path.read_text())
                domain = "无延迟" if deployment == "clean" else "50 ms 延迟"
                title = f"{label} · 种子 {model['training_seed']} | {domain} | 场景 {seed}"
                catalog.append(render(Path(row["trajectory"]), name=name, title=title,
                                      category="learned_policy_autonomous_rollout", success=row["success"]))
    seed = protocol["development_seeds"][0]
    for deployment, directory in (("clean", "clean__cleanof-D2"), ("D2", "D2__cleanof-none")):
        source = ROOT / f"runtime_results/engineering/learning-20261025-v1/causal-data/datasets/Stack/Panda/{directory}/L1/seed-{seed}.hdf5"
        with h5py.File(source, "r") as handle:
            success = int(handle["data/demo_0"].attrs["t_success"]) >= 0
        domain = "无延迟" if deployment == "clean" else "50 ms 延迟"
        catalog.append(render(source, name=f"teacher-{deployment}-development-{seed}",
            title=f"教师示教（历史轨迹） | {domain} | 场景 {seed}",
            category="historical_teacher_demonstration", success=success))
    MEDIA.mkdir(exist_ok=True)
    (MEDIA / "catalog.json").write_text(json.dumps({"selection": "first development scene and first new evaluation scene, fixed by protocol order",
        "items": catalog, "pending": pending}, ensure_ascii=False, indent=2) + "\n")
    if pending and not args.available_only:
        raise ValueError(f"{len(pending)} requested trajectories have not completed")
    print(json.dumps({"videos": len(catalog), "pending": len(pending)}))


if __name__ == "__main__":
    main()
