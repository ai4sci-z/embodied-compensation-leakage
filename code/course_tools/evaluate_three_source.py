#!/usr/bin/env python3
"""Measure source-compensation inheritance and execute an autonomous target episode.

Fixed-state mode uses the manifest test split. A source's own same-state normal
teacher is the reference. A label intervention is reported separately from the
leakage coefficient. Rollout mode executes one policy/target/scene, without a
teacher, and stores full physical and image evidence for later paired analysis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "dl_course.three_source_evaluation/2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                              allow_nan=False) + "\n")
    tmp.replace(path)


def field_metrics(error, compensation, *, energy_floor=1e-12):
    """L2 fields: ratio of sums, never a mean of framewise ratios."""
    import numpy as np
    e, c = np.asarray(error, float), np.asarray(compensation, float)
    if e.shape != c.shape or e.ndim != 2 or not np.isfinite(e).all() or not np.isfinite(c).all():
        raise ValueError("fields must be matching finite N x D arrays")
    if len(e) == 0:
        return {"status": "empty", "frames": 0, "lambda": None, "phi": None,
                "direction_cosine": None, "rmse": None}
    dot = float(np.sum(e * c))
    ce = float(np.sum(c * c))
    ee = float(np.sum(e * e))
    result = {"status": "defined", "frames": len(e), "dimensions": e.shape[1],
        "dot_sum": dot, "compensation_energy_sum": ce, "error_energy_sum": ee,
        "mean_compensation_norm_squared": ce / len(e),
        "mean_error_norm_squared": ee / len(e), "rmse": float(np.sqrt(ee / e.size)),
        "lambda": None, "phi": None, "direction_cosine": None,
        "compensation_rms": float(np.sqrt(ce / c.size)),
        "energy_floor_per_frame": energy_floor}
    if ce / len(e) <= energy_floor:
        result["status"] = "undefined_near_zero_compensation"
        return result
    result["lambda"] = dot / ce
    if ee / len(e) <= energy_floor:
        result["status"] = "zero_error_direction_undefined"
        return result
    result["direction_cosine"] = dot / float(np.sqrt(ee * ce))
    result["phi"] = min(1., max(0., dot * dot / (ee * ce)))
    return result


def smooth_reference(value, window=5):
    import numpy as np
    padded = np.pad(np.asarray(value), ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window].mean(0) for i in range(len(value))])


def adjusted_metrics(error, compensation, nuisance, *, energy_floor=1e-12):
    """Partial coefficient with an explicit rank/residual-energy diagnostic."""
    import numpy as np
    e, c = np.asarray(error, float), np.asarray(compensation, float)
    if len(e) == 0:
        return {"status": "empty", "lambda_partial": None}
    columns = np.column_stack([np.asarray(x, float).ravel() for x in nuisance.values()])
    norms = np.linalg.norm(columns, axis=0)
    active = norms > 1e-12
    z = columns[:, active] / norms[active] if active.any() else np.empty((e.size, 0))
    rank = int(np.linalg.matrix_rank(z)) if z.shape[1] else 0
    singular = np.linalg.svd(z, compute_uv=False) if z.shape[1] else np.array([])
    cond = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 1e-14 else None
    flat_c, flat_e = c.ravel(), e.ravel()
    cr = flat_c - z @ np.linalg.lstsq(z, flat_c, rcond=None)[0]
    er = flat_e - z @ np.linalg.lstsq(z, flat_e, rcond=None)[0]
    ce, residual_energy = float(flat_c @ flat_c), float(cr @ cr)
    share = residual_energy / ce if ce > 0 else None
    out = {"status": "defined", "lambda_partial": None,
        "nuisance_names": list(nuisance), "nuisance_rank": rank,
        "active_nuisance_columns": int(active.sum()), "nuisance_condition": cond,
        "nuisance_rank_deficient": rank < int(active.sum()),
        "compensation_residual_share": share,
        "vif": 1 / share if share is not None and share > 0 else None,
        "nuisance_adjustment_note": "Reference shrinkage, centered five-frame smoothing, and gain fields; not an automatic causal separation."}
    if ce / len(e) <= energy_floor or residual_energy / len(e) <= energy_floor or share <= 1e-8:
        out["status"] = "undefined_compensation_in_nuisance_span"
        return out
    out["lambda_partial"] = float(cr @ er / residual_energy)
    out["residual_error_energy"] = float(er @ er)
    return out


def direction_controls(compensation):
    """Eight fixed norm-preserving rotations; these are not physical defect truth."""
    import numpy as np
    c = np.asarray(compensation, float)
    out = []
    angles = ((0, 90), (0, -90), (1, 90), (1, -90),
              (2, 90), (2, -90), (0, 180), (1, 180))
    for axis, degree in angles:
        angle = np.deg2rad(degree)
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.eye(3)
        i, j = [k for k in range(3) if k != axis]
        rotation[i, i] = rotation[j, j] = co
        rotation[i, j], rotation[j, i] = -si, si
        transformed = np.concatenate([c[:, start:start + 3] @ rotation.T
                                      for start in range(0, c.shape[1], 3)], axis=1)
        out.append((f"axis{axis}_{degree:+d}deg", transformed))
    return out


def component_readings(prediction, clean, compensation, train_mean, masks):
    import numpy as np
    # Translation and rotation are separate physical command coordinates;
    # the canonical 6D reading is supplementary because it mixes units.
    components = {"translation_command_m": (slice(0, 3), .05, 2.5e-15),
                  "rotation_command_rad": (slice(3, 6), .5, 2.5e-13),
                  "canonical_action_6d": (slice(0, 6), 1., 1e-12)}
    result = {}
    for name, (sl, scale, floor) in components.items():
        e = (prediction[:, sl] - clean[:, sl]) * scale
        c = compensation[:, sl] * scale
        ref = clean[:, sl] * scale
        nuisance = {"shrinkage": train_mean[sl] * scale - ref,
                    "time_smoothing": smooth_reference(ref) - ref, "gain": ref}
        strata = {}
        for label, mask in masks.items():
            reading = field_metrics(e[mask], c[mask], energy_floor=floor)
            if label == "all":
                reading["partial"] = adjusted_metrics(e, c, nuisance, energy_floor=floor)
                controls = []
                for control, surrogate in direction_controls(c):
                    controls.append({"name": control, "raw": field_metrics(e, surrogate, energy_floor=floor),
                        "partial": adjusted_metrics(e, surrogate, nuisance, energy_floor=floor),
                        "cosine_to_true_field": field_metrics(c, surrogate, energy_floor=floor)["direction_cosine"],
                        "identical_to_true_field": bool(np.allclose(c, surrogate, rtol=1e-10, atol=1e-14))})
                reading["direction_controls"] = controls
                values = [r["raw"]["lambda"] for r in controls if r["raw"]["lambda"] is not None]
                reading["real_minus_mean_control_projection"] = (
                    reading["lambda"] - float(np.mean(values))
                    if reading["lambda"] is not None and len(values) == len(controls) else None)
                reading["control_interpretation"] = (
                    "Directional specificity only. Rotations preserve frame norms and support, "
                    "but are not independently realized physical defects or a permutation null.")
            strata[label] = reading
        result[name] = strata
    return result


def intervention_readings(raw, removed, compensation, masks):
    out = {}
    for name, sl, scale, floor in (
        ("translation_command_m", slice(0, 3), .05, 2.5e-15),
        ("rotation_command_rad", slice(3, 6), .5, 2.5e-13),
        ("canonical_action_6d", slice(0, 6), 1., 1e-12),
    ):
        out[name] = {}
        for label, mask in masks.items():
            row = field_metrics((raw[:, sl] - removed[:, sl])[mask] * scale,
                                compensation[:, sl][mask] * scale, energy_floor=floor)
            row["intervention_projection_coefficient"] = row.pop("lambda")
            row["intervention_direction_energy_share"] = row.pop("phi")
            out[name][label] = row
    return out


def train_action_mean(manifest, config):
    import h5py
    import numpy as np
    source_scenes = defaultdict(lambda: defaultdict(list))
    for row in manifest.rows:
        if row["split"] != "train":
            continue
        if config.get("level") is not None and row["source"] != "clean" and row["level"] != config["level"]:
            continue
        with h5py.File(row["path"], "r") as h:
            a = h[f"data/{row['demo']}/actions"][:, :6]
        source_scenes[row["source"]][row["seed_env"]].append(a)
    total = np.zeros(6)
    for source, weight in config["source_weights"].items():
        if weight <= 0:
            continue
        if not source_scenes[source]:
            raise ValueError(f"no original training labels for active source {source}")
        scene_means = [np.concatenate(arrays).mean(0) for arrays in source_scenes[source].values()]
        total += weight * np.stack(scene_means).mean(0)
    return total


def frame_masks(group):
    import numpy as np
    n = len(group["actions"])
    masks = {"all": np.ones(n, dtype=bool)}
    if "phase" in group:
        phase = group["phase"][:]
        masks.update({f"phase_{int(k)}": phase == k for k in np.unique(phase)})
    if "mechanism_features" in group:
        f = group["mechanism_features"]
        active = f["active_joint_indices"][:].astype(int)
        reversal = f["joint_reversal"][:, active].any(axis=1)
        masks["at_active_joint_reversal"] = reversal
        masks["within_three_steps_after_reversal"] = np.convolve(reversal, np.ones(4), mode="full")[:n] > 0
        masks["active_joint_disengaged"] = (~f["engaged"][:, active].astype(bool)).any(axis=1)
        speed = f["eef_speed_m_s"][:]
        masks["moving_gt_1mm_s"] = speed > .001
        masks["stationary_le_1mm_s"] = speed <= .001
        masks["speed_1_to_20mm_s"] = (speed > .001) & (speed <= .020)
        masks["speed_gt_20mm_s"] = speed > .020
        gravity_norm = np.linalg.norm(f["gravity_torque_nm"][:], axis=1)
        median = np.median(gravity_norm)
        masks["gravity_norm_le_episode_median"] = gravity_norm <= median
        masks["gravity_norm_gt_episode_median"] = gravity_norm > median
    return masks


def verify_contrast(primary, contrast):
    if primary.metadata["config"]["label_mode"] != "raw":
        raise ValueError("label intervention primary checkpoint must have raw labels")
    if contrast.metadata["config"]["label_mode"] == "raw":
        raise ValueError("contrast checkpoint must remove compensation labels")
    a = {k: v for k, v in primary.metadata["config"].items() if k != "label_mode"}
    b = {k: v for k, v in contrast.metadata["config"].items() if k != "label_mode"}
    if a != b:
        raise ValueError("label contrast changed more than its label condition")
    for key in ("step", "sampling_chain_sha256", "source_sample_counts"):
        if primary.metadata[key] != contrast.metadata[key]:
            raise ValueError(f"label contrast does not match {key}")
    if primary.statistics != contrast.statistics:
        raise ValueError("label contrast used different normalization")


def fixed_state(args):
    import h5py
    import numpy as np
    import torch
    from course_tools.three_source_data import EpisodeManifest, PoolDataset
    from course_tools.train_three_source_pool import load_three_source_checkpoint

    torch.set_num_threads(1)
    manifest = EpisodeManifest.load(args.manifest)
    loaded = load_three_source_checkpoint(args.checkpoint, device=args.device)
    if manifest.manifest_sha256 != loaded.metadata["run_identity"]["manifest_sha256"]:
        raise ValueError("evaluation manifest differs from the checkpoint data manifest")
    dataset = PoolDataset(manifest, "test", k_hist=loaded.history, cache_mib=args.cache_mib)
    contrast = load_three_source_checkpoint(args.contrast_checkpoint, device=args.device) if args.contrast_checkpoint else None
    if contrast is not None:
        verify_contrast(loaded, contrast)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": SCHEMA, "mode": "fixed_state", "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_step": loaded.metadata["step"],
        "config": loaded.metadata["config"], "runner_sha256": sha256(__file__),
        "contrast_sha256": sha256(args.contrast_checkpoint) if contrast else None,
        "test_sources": sorted({r["source"] for r in dataset.rows}),
        "smoothing_window": 5, "reference": "source-paired same-state normal teacher",
        "statistical_unit": "scene; frame counts describe fields, not independent replicates"}
    protocol["protocol_sha256"] = hash_json(protocol)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("output contains a different fixed-state experiment")
    atomic_json(protocol_path, protocol)
    mean = train_action_mean(manifest, loaded.metadata["config"])
    indices = defaultdict(list)
    for i in range(len(dataset.raw)):
        identity, step = dataset.raw.window_identity(i)
        indices[identity].append((step, i))
    reports = []
    for row in dataset.rows:
        stem = f"{row['source']}-L{row['level']}-seed-{row['seed_env']}-{hash_json(row['id'])[:8]}"
        report_path = out / "episodes" / f"{stem}.json"
        array_path = out / "predictions" / f"{stem}.hdf5"
        if report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved["protocol_sha256"] != protocol["protocol_sha256"] or sha256(array_path) != saved["predictions_sha256"]:
                raise ValueError(f"saved fixed-state episode changed: {stem}")
            reports.append(saved)
            continue
        ordered = sorted(indices[row["id"]])
        if [x[0] for x in ordered] != list(range(row["horizon"])):
            raise ValueError("test episode does not cover each control step exactly once")
        pred, alt = [], []
        for start in range(0, len(ordered), args.batch_size):
            batch, _ = dataset.batch([x[1] for x in ordered[start:start + args.batch_size]])
            pred.append(loaded.predict(batch.obs, clip=False)[:, 0])
            if contrast:
                alt.append(contrast.predict(batch.obs, clip=False)[:, 0])
        prediction = np.concatenate(pred)
        removed = np.concatenate(alt) if contrast else None
        with h5py.File(row["path"], "r") as source:
            group = source[f"data/{row['demo']}"]
            clean, c = group["actions_clean"][:], group["c_true"][:]
            if not np.allclose(group["actions"][:] - clean, c, atol=3e-7):
                raise ValueError("saved source actions do not decompose into normal teacher plus c_true")
            masks = frame_masks(group)
            report = {"schema": SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
                "source": row["source"], "seed_env": row["seed_env"], "level": row["level"],
                "episode_id": row["id"], "reference": "same-state paired normal teacher",
                "normal_reference_is_trained_oracle": False,
                "statistics_unit": "one scene/source/level", "frames": len(prediction),
                "clipped": component_readings(np.clip(prediction, -1, 1), clean, c, mean, masks),
                "unclipped": component_readings(prediction, clean, c, mean, masks),
                "continuous_saturation_fraction": float((np.abs(prediction[:, :6]) >= 1).mean())}
            if contrast:
                report["label_intervention"] = {
                    "contrast_mode": contrast.metadata["config"]["label_mode"],
                    "sampling_chain_verified_equal": True,
                    "clipped": intervention_readings(np.clip(prediction, -1, 1), np.clip(removed, -1, 1), c, masks),
                    "unclipped": intervention_readings(prediction, removed, c, masks)}
            array_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = array_path.with_suffix(f".{os.getpid()}.partial.hdf5")
            with h5py.File(temporary, "w") as target:
                target.attrs["metadata_json"] = json.dumps({k: report[k] for k in (
                    "schema", "protocol_sha256", "source", "seed_env", "level", "episode_id")})
                for key, value in {"prediction_unclipped": prediction,
                    "prediction_clipped": np.clip(prediction, -1, 1),
                    "actions_clean": clean, "c_true": c,
                    "control_step": np.arange(len(prediction))}.items():
                    target.create_dataset(key, data=value, compression="gzip")
                if contrast:
                    target.create_dataset("contrast_prediction_unclipped", data=removed, compression="gzip")
                    target.create_dataset("contrast_prediction_clipped", data=np.clip(removed, -1, 1), compression="gzip")
                for name in ("phase", "mechanism_features", "reference_trace"):
                    if name in group:
                        group.copy(name, target)
            temporary.replace(array_path)
        report.update(predictions=str(array_path), predictions_sha256=sha256(array_path))
        atomic_json(report_path, report)
        reports.append(report)
        atomic_json(out / "progress.json", {"status": "running", "completed": len(reports), "planned": len(dataset.rows)})
    summary = {"schema": SCHEMA, "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "episode_count": len(reports), "unique_scene_count": len({x["seed_env"] for x in reports}),
        "episode_reports": [str(p) for p in sorted((out / "episodes").glob("*.json"))],
        "grouped_scene_means": summarize_readings(reports),
        "note": "Means use one vote per episode within a source/level. Shared scenes and model seeds remain paired; no framewise significance test."}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", summary)
    return summary


def summarize_readings(reports):
    import numpy as np
    buckets = defaultdict(list)
    for row in reports:
        buckets[(row["source"], row["level"])].append(row)
    result = []
    for (source, level), rows in sorted(buckets.items()):
        readings = {}
        for component in ("translation_command_m", "rotation_command_rad", "canonical_action_6d"):
            readings[component] = {}
            for metric in ("lambda", "phi", "direction_cosine", "rmse"):
                values = [x["clipped"][component]["all"][metric] for x in rows]
                valid = [x for x in values if x is not None]
                readings[component][metric] = {"mean": float(np.mean(valid)) if len(valid) == len(rows) else None,
                    "defined_scenes": len(valid), "total_scenes": len(rows)}
        result.append({"source": source, "level": level, "scenes": len(rows), "readings": readings})
    return result


def make_target_pair(target, level, task, seed):
    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import DefectWrapper, TwinPair
    from portal_trace.teacher import osc
    from portal_trace.types import SeedTriple

    code = "D2" if target == "normal" else target
    actual_level = 0 if target == "normal" else level
    delta = Delta.from_level(code, actual_level, joint_scale=default_joint_scale(code, 7))
    bodies = []
    try:
        for value in (delta, delta.zero()):
            env = osc.make_env(task, family="Panda", seed_env=seed, render_offscreen=False)
            if code == "D4hi":
                from course_tools.backlash_physics import make_wrapper
                wrapper = make_wrapper
            else:
                wrapper = DefectWrapper
            bodies.append(wrapper(env, value, seeds=SeedTriple(env=seed, train=-1, eval=0), family="Panda"))
        pair = TwinPair(defective=bodies[0], clean=bodies[1], delta=delta,
                        family="Panda", task=task, extra_nodes=bodies[0].index.shadow_names)
        osc.reset_twin_pair(pair, seed)
        return pair
    except BaseException:
        for body in bodies:
            body.close()
        raise


def target_physics_export(body):
    """Identify the actual deployed physical model, including injected joints."""
    if body.delta.defect_code == "D4hi":
        from course_tools.backlash_physics import PHYSICS_VERSION
        if getattr(body, "physics_version", None) != PHYSICS_VERSION:
            raise ValueError("D4hi deployment must use the recorded unilateral-limit physics")
        return body.export_physics()
    from course_tools.backlash_physics import model_state_map
    xml = body.sim.model.get_xml()
    return {"model_xml": xml, "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "physics_version": "portal_trace.DefectWrapper." + body.delta.defect_code,
            "state_map": model_state_map(body.sim.model._model),
            "defect_state_contract": "portal_trace.WrapperState.flat"}


def make_deployment_adapter(body, history):
    from portal_trace.policies.obs import PROPRIO_ACCELERATION_CAUSAL
    from portal_trace.runners.robosuite_adapter import RobosuiteObservationAdapter
    from portal_trace.teacher import osc
    return RobosuiteObservationAdapter(env=body,
        aux_provider=lambda raw, *, t: (body.gravity_torque(), osc.wrist_ft(body)),
        task_index=0, embodiment_index=0, k_hist=history,
        acceleration_semantics=PROPRIO_ACCELERATION_CAUSAL)


def deployment_frame(body, adapter, step):
    """Build the actual network input and assert its physical coordinate order."""
    import mujoco
    import numpy as np
    body.sim.forward()
    raw = body.env._get_observations(force_update=True)
    # Use one renderer bound to the final live model. robosuite 1.5.2 twin
    # resets can leave stale contexts/camera buffers; native Renderer owns and
    # activates its context on every call. Its RGB is top-down, so expose the
    # raw bottom-up convention expected by RobosuiteObservationAdapter.
    if not hasattr(adapter, "_course_renderer"):
        adapter._course_renderer = mujoco.Renderer(body.sim.model._model, height=84, width=84)
        adapter._course_render_options = mujoco.MjvOption()
        adapter._course_render_options.geomgroup[0] = 0
        adapter._course_render_options.geomgroup[1] = 1
    for camera in ("agentview", "robot0_eye_in_hand"):
        adapter._course_renderer.update_scene(body.sim.data._data, camera=camera,
                                              scene_option=adapter._course_render_options)
        raw[camera + "_image"] = adapter._course_renderer.render()[::-1].copy()
    for key, expected in (
        ("robot0_joint_pos", body.sim.data.qpos[list(body.index.qpos_index)]),
        ("robot0_joint_vel", body.sim.data.qvel[list(body.index.qvel_index)]),
    ):
        if not np.allclose(raw[key], expected, atol=1e-8, rtol=0):
            raise ValueError(f"deployed {key} differs from named physical coordinates")
    observation = adapter.to_observation(raw, t=step)
    # The three-source policy is intentionally unconditioned on source identity.
    observation.embodiment_index = None
    observation.partition_index = None
    if tuple(observation.proprio.shape) != (1, adapter.k_hist, 70):
        raise ValueError("deployment proprioception differs from the training input")
    if not np.isfinite(observation.proprio.numpy()).all():
        raise ValueError("deployment proprioception is nonfinite")
    return raw, observation


def verify_target_interface(args):
    """Exercise live physics/observation I/O without loading any policy."""
    import h5py
    import numpy as np
    from portal_trace.teacher import osc
    from PIL import Image

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair = make_target_pair(args.target, args.level, args.task, args.seed_env)
    body = pair.clean if args.target == "normal" else pair.defective
    try:
        export = target_physics_export(body)
        adapter = make_deployment_adapter(body, 5)
        props, states, pixels = [], [], []
        for step in range(3):
            raw, obs = deployment_frame(body, adapter, step)
            states.append(osc.physics_state(body.sim))
            props.append(obs.proprio[0, -1].numpy())
            pixels.append({key: np.asarray(raw[key])[::-1].copy() for key in obs.images})
            hold = np.zeros(7)
            hold[-1] = -1
            body.step(osc.env_action(body, hold))
        name = f"{args.target}-L{args.level}-seed-{args.seed_env}"
        trace = output / f"{name}.hdf5"
        with h5py.File(trace, "x") as handle:
            handle.attrs["purpose"] = "environment and observation interface only; no trained policy evaluated"
            handle.attrs["model_xml"] = export["model_xml"]
            handle.create_dataset("states", data=np.stack(states))
            handle.create_dataset("proprio", data=np.stack(props))
            for camera in pixels[0]:
                handle.create_dataset(camera, data=np.stack([frame[camera] for frame in pixels]))
                Image.fromarray(pixels[0][camera]).resize((336, 336)).save(output / f"{name}-{camera}.png")
        checks = {"finite_proprio": True, "proprio_shape": list(np.stack(props).shape),
                  "history_shape": list(obs.proprio.shape), "nq": int(body.sim.model.nq),
                  "nv": int(body.sim.model.nv), "source_identity_hidden": obs.embodiment_index is None}
        if args.target == "D4hi":
            equality = np.asarray(body.sim.data.eq_active)[list(body.index.shadow_equality_ids)]
            checks["shadow_equalities_disabled"] = bool(not equality.any())
            if equality.any():
                raise ValueError("legacy backlash equality remains active")
            checks["shadow_joints"] = list(body.index.shadow_names)
            checks["transmission"] = body.backlash_diagnostics()
        row = {"schema": SCHEMA, "mode": "environment_interface_verification",
               "target": args.target, "level": args.level, "seed_env": args.seed_env,
               "policy_loaded": False, "policy_evaluation_performed": False,
               "physics": {k: v for k, v in export.items() if k != "model_xml"},
               "checks": checks, "trace": str(trace), "trace_sha256": sha256(trace),
               "runner_sha256": sha256(__file__)}
        atomic_json(output / f"{name}.json", row)
        return row
    finally:
        if "adapter" in locals() and hasattr(adapter, "_course_renderer"):
            adapter._course_renderer.close()
        pair.defective.close()
        pair.clean.close()


def rollout(args):
    import h5py
    import mujoco
    import numpy as np
    import torch
    from course_tools.train_three_source_pool import load_three_source_checkpoint
    from portal_trace.success import terminal_hold_start
    from portal_trace.teacher import osc

    if args.task != "Stack":
        raise ValueError("this rollout stores Stack-specific object measures; other tasks need explicit outcome definitions")
    torch.set_num_threads(1)
    policy = load_three_source_checkpoint(args.checkpoint, device=args.device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ck_sha = sha256(args.checkpoint)
    stem = f"{ck_sha[:12]}-{args.target}-L{args.level}-seed-{args.seed_env}"
    summary_path, trajectory_path = out / "summaries" / f"{stem}.json", out / "trajectories" / f"{stem}.hdf5"
    protocol = {"schema": SCHEMA, "mode": "autonomous_rollout", "checkpoint_sha256": ck_sha,
        "target": args.target, "level": args.level, "task": args.task, "seed_env": args.seed_env,
        "horizon": 400, "execution_horizon": 1, "control_hz": 20, "rgb_stride": 1,
        "runner_sha256": sha256(__file__), "policy_config": policy.metadata["config"],
        "physics_code_sha256": {
            "wrapper": sha256(ROOT / "portal_trace/defects/inject.py"),
            "backlash": sha256(ROOT / "course_tools/backlash_physics.py")},
        "checkpoint_training_manifest_sha256": policy.metadata["run_identity"]["manifest_sha256"],
        "normal_reference": "D2.zero; D4hi unilateral-limit target uses its dimension-compatible clean twin for initialization"}
    protocol_sha = hash_json(protocol)
    if summary_path.exists():
        row = json.loads(summary_path.read_text())
        if row["protocol_sha256"] != protocol_sha or sha256(trajectory_path) != row["trajectory_sha256"]:
            raise ValueError("existing trajectory differs from requested evaluation")
        return row
    atomic_json(out / "protocols" / f"{stem}.json", protocol)
    status = out / "workers" / f"{stem}.json"
    atomic_json(status, {"status": "starting", "pid": os.getpid(), "protocol_sha256": protocol_sha})
    pair = make_target_pair(args.target, args.level, args.task, args.seed_env)
    body = pair.clean if args.target == "normal" else pair.defective
    try:
        adapter = make_deployment_adapter(body, policy.history)
        policy.reset_episode()
        T = 400
        states = np.empty((T + 1, osc.physics_state(body.sim).size))
        defect_state = np.empty((T + 1, body.defect_state().size))
        actuation = np.empty((T + 1, body.actuation_state().size))
        actions, unclipped = np.empty((T, 7)), np.empty((T, 7))
        applied = np.empty((T, body.action_dim))
        effective = np.empty_like(applied)
        torque = np.empty((T, 7))
        objects = np.empty((T + 1, 2, 3))
        eef = np.empty((T + 1, 3))
        joint_q = np.empty((T + 1, 7))
        joint_v = np.empty_like(joint_q)
        gravity = np.empty((T, 7))
        wrist = np.empty((T, 6))
        force = np.empty(T)
        contact_count = np.empty(T, dtype=int)
        success = np.zeros(T, dtype=bool)
        rewards = np.empty(T)
        dones = np.empty(T, dtype=bool)
        task_contact = np.zeros(T, dtype=bool)
        proprio = np.empty((T, 70), dtype=np.float32)
        transmission_rows = []
        rgb = {k: [] for k in ("agentview_image", "robot0_eye_in_hand_image")}
        cube_ids = [int(body.sim.model.body_name2id(k)) for k in ("cubeA_main", "cubeB_main")]
        grip = osc.OSCTracker().grip_site_id(body)
        qidx, vidx = list(body.index.qpos_index), list(body.index.qvel_index)
        physics = target_physics_export(body)
        xml = physics["model_xml"]
        from portal_trace.registry.tasks import task as task_spec
        moving_geoms, target_geoms = osc.mating_geom_ids(body, task_spec(args.task))
        initial_object_quaternions = np.asarray(body.sim.data.body_xquat[cube_ids]).copy()
        started = time.monotonic()
        for t in range(T + 1):
            body.sim.forward()
            states[t] = osc.physics_state(body.sim)
            defect_state[t] = body.defect_state()
            actuation[t] = body.actuation_state()
            objects[t] = body.sim.data.body_xpos[cube_ids]
            eef[t] = body.sim.data.site_xpos[grip]
            joint_q[t], joint_v[t] = body.sim.data.qpos[qidx], body.sim.data.qvel[vidx]
            if args.target == "D4hi":
                transmission_rows.append(body.backlash_diagnostics())
            if t == T:
                break
            raw, observation = deployment_frame(body, adapter, t)
            gravity[t], wrist[t] = body.gravity_torque(), osc.wrist_ft(body)
            for camera in rgb:
                rgb[camera].append(np.asarray(raw[camera])[::-1].copy())
            proprio[t] = observation.proprio[0, -1].numpy()
            unclipped[t] = policy.predict(observation, clip=False)[0, 0]
            if not np.isfinite(unclipped[t]).all():
                raise ValueError(f"nonfinite policy action at step {t}")
            actions[t] = np.clip(unclipped[t], -1, 1)
            applied[t] = osc.env_action(body, actions[t])
            _, rewards[t], dones[t], _ = body.step(applied[t])
            effective[t] = body.state.effective_action
            torque[t] = body.commanded_joint_torque()
            mujoco.mj_rnePostConstraint(body.sim.model._model, body.sim.data._data)
            force[t] = np.linalg.norm(np.asarray(body.sim.data.cfrc_ext)[:, 3:6], axis=1).max()
            contact_count[t] = int(body.sim.data.ncon)
            task_contact[t] = osc._touching(body, moving_geoms, target_geoms)
            success[t] = bool(body.env._check_success())
            if (t + 1) % 50 == 0:
                atomic_json(status, {"status": "running", "pid": os.getpid(), "step": t + 1,
                    "updated_unix": time.time(), "protocol_sha256": protocol_sha})
        terminal = int(terminal_hold_start(success))
        row = {"schema": SCHEMA, "protocol_sha256": protocol_sha, "checkpoint_sha256": ck_sha,
            "training_seed": policy.metadata["config"]["seed"],
            "label_mode": policy.metadata["config"]["label_mode"],
            "source_weights": policy.metadata["config"]["source_weights"],
            "task": args.task, "target": args.target, "target_delta": body.delta.to_json(),
            "physics": {k: v for k, v in physics.items() if k != "model_xml"},
            "level": args.level, "seed_env": args.seed_env,
            "success": terminal >= 0, "terminal_hold_start": terminal,
            "any_success": bool(success.any()), "terminal_success_frames": int(success[-20:].sum()),
            "cube_a_max_lift_m": float((objects[:, 0, 2] - objects[0, 0, 2]).max()),
            "terminal_cube_xy_distance_m": float(np.linalg.norm(objects[-1, 0, :2] - objects[-1, 1, :2])),
            "minimum_eef_cube_a_distance_m": float(np.linalg.norm(eef - objects[:, 0], axis=1).min()),
            "peak_control_boundary_force_n": float(force.max()),
            "force_measurement": "20 Hz post-action body cfrc_ext; not the maximum over 500 Hz substeps",
            "continuous_action_saturation_fraction": float((np.abs(unclipped[:, :6]) >= 1).mean()),
            "initial_physics_sha256": hashlib.sha256(states[0].tobytes()).hexdigest(),
            "initial_gripper_sha256": hashlib.sha256(actuation[0].tobytes()).hexdigest(),
            "initial_named_joint_qpos": dict(zip(body.index.joint_names, joint_q[0].tolist())),
            "initial_named_joint_qvel": dict(zip(body.index.joint_names, joint_v[0].tolist())),
            "initial_cube_positions": objects[0].tolist(),
            "initial_cube_quaternions_wxyz": initial_object_quaternions.tolist(),
            "state_dimensions": {"nq": int(body.sim.model.nq), "nv": int(body.sim.model.nv)},
            "pairing_note": "Compare initial named joints and object poses across targets; bytewise state hashes apply only within the same model structure.",
            "teacher_intervention": False, "policy_object_state_input": False,
            "observation_history": policy.history, "rgb_stride": 1,
            "rgb_orientation": "vertically flipped raw MuJoCo image, matching training",
            "wall_seconds": time.monotonic() - started}
        arrays = {"states_pre_action_plus_terminal": states, "defect_state_pre_action_plus_terminal": defect_state,
            "gripper_accumulator_pre_action_plus_terminal": actuation, "actions": actions,
            "actions_unclipped": unclipped, "applied_actions": applied,
            "effective_actions_after_delay": effective, "commanded_joint_torque_nm": torque,
            "object_positions_pre_action_plus_terminal": objects, "eef_position_pre_action_plus_terminal": eef,
            "joint_qpos_pre_action_plus_terminal": joint_q, "joint_qvel_pre_action_plus_terminal": joint_v,
            "gravity_torque_pre_action": gravity, "wrist_ft_pre_action": wrist,
            "external_force_n_post_action": force, "contact_count_post_action": contact_count,
            "success_post_action": success, "rgb_control_step": np.arange(T),
            "rewards_post_action": rewards, "dones_post_action": dones,
            "task_contact_post_action": task_contact, "policy_proprio_pre_action": proprio,
            **{k: np.stack(v) for k, v in rgb.items()}}
        if transmission_rows:
            for name in ("input_position_rad", "output_position_rad", "relative_position_rad",
                         "input_velocity_rad_s", "output_velocity_rad_s", "relative_velocity_rad_s",
                         "limit_active", "boundary_side", "transmitting", "transmission_torque_nm",
                         "penetration_rad", "total_width_rad"):
                arrays["transmission_" + name + "_pre_action_plus_terminal"] = np.asarray([
                    [transmission[name] for transmission in row] for row in transmission_rows])
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("nonfinite saved rollout")
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = trajectory_path.with_suffix(f".{os.getpid()}.partial.hdf5")
        with h5py.File(tmp, "w") as h:
            h.attrs["metadata_json"] = json.dumps(row, sort_keys=True)
            h.attrs["model_xml"] = xml
            for name, value in arrays.items():
                h.create_dataset(name, data=value, compression="lzf")
        tmp.replace(trajectory_path)
        row.update(trajectory=str(trajectory_path), trajectory_sha256=sha256(trajectory_path))
        atomic_json(summary_path, row)
        atomic_json(status, {"status": "complete", "summary": str(summary_path), "protocol_sha256": protocol_sha})
        return row
    except BaseException as exc:
        atomic_json(status, {"status": "failed", "error": repr(exc),
                            "protocol_sha256": protocol_sha, "updated_unix": time.time()})
        raise
    finally:
        if "adapter" in locals() and hasattr(adapter, "_course_renderer"):
            adapter._course_renderer.close()
        pair.defective.close()
        pair.clean.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--contrast-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--verify-environment", action="store_true",
                        help="record three live I/O frames; no policy is loaded or evaluated")
    parser.add_argument("--target", choices=("normal", "D2", "D4hi", "D3b"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed-env", type=int)
    parser.add_argument("--task", default="Stack")
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cache_mib < 0:
        parser.error("invalid inference batch/cache setting")
    if args.rollout or args.verify_environment:
        if args.target is None or args.seed_env is None:
            parser.error("live environment mode requires --target and --seed-env")
        if args.contrast_checkpoint:
            parser.error("execute label-contrast checkpoints as separate paired rollouts")
    elif args.manifest is None:
        parser.error("fixed-state mode requires --manifest")
    if args.verify_environment and (args.rollout or args.checkpoint):
        parser.error("environment verification does not accept a policy checkpoint or rollout mode")
    if not args.verify_environment and args.checkpoint is None:
        parser.error("model evaluation requires --checkpoint")
    if args.target != "normal" and args.level not in (1, 2):
        parser.error("defective target level must be 1 or 2")
    os.environ["MUJOCO_GL"] = args.render_backend
    if args.render_backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.setdefault("LP_NUM_THREADS", "1")
    result = verify_target_interface(args) if args.verify_environment else rollout(args) if args.rollout else fixed_state(args)
    print(json.dumps({"status": "complete", "mode": result.get("mode", "rollout" if args.rollout else "fixed_state"),
                      "output": str(args.output.resolve()),
                      "success": result.get("success"), "episodes": result.get("episode_count")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
