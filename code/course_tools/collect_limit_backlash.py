#!/usr/bin/env python3
"""Collect a paired teacher episode with explicitly versioned mechanical backlash.

Example (inside the existing course container)::

    python course_tools/collect_three_source.py --defect D4hi --level 1 \
      --seed-env 123 --split train --threshold-report /path/regimes.json \
      --tol 0.02 --output-root course_results/three-source-data --materialise

The teacher runs without image rendering. Raw files retain same-state teacher
queries, all replay state, reference poses, and ILC histories. ``--materialise``
uses the existing robomimic conversion and causal proprioception pipeline.
An unsuccessful teacher episode is stored and identified, never silently retried
until success or promoted into a successful demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {"D2": "delay", "D4hi": "backlash", "D4lo": "backlash", "D3b": "droop"}
SCHEMA = "dl_course.limit_backlash_collection/2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def config_payload(config):
    return {"lead_steps": config.lead_steps, "q_cutoff_hz": config.q_cutoff_hz,
            "k_max": config.k_max, "contact_variant": config.contact_variant.value}


def make_pair(task, family, defect, level, seed_env):
    from course_tools.backlash_physics import make_pair as limit_pair
    if defect != "D4hi":
        raise ValueError("this collector is only for versioned mechanical backlash")
    return limit_pair(task, family, level, seed_env)


def mechanism_features(tracking, wrapper, reference):
    import numpy as np
    from portal_trace.defects.inject import ZERO_VELOCITY_BAND
    from course_tools.limit_backlash_data import transmission_arrays
    from portal_trace.registry.tasks import CONTROL_FREQ

    nq = int(wrapper.sim.model.nq)
    q = tracking.states[:, np.asarray(wrapper.index.qpos_index)]
    v = tracking.states[:, nq + np.asarray(wrapper.index.qvel_index)]
    shadow_q = tracking.states[:, np.asarray(wrapper.index.shadow_qpos_index)]
    shadow_v = tracking.states[:, nq + np.asarray(wrapper.index.shadow_qvel_index)]
    direction = np.where(np.abs(v) > ZERO_VELOCITY_BAND, np.sign(v), 0)
    last = np.zeros(v.shape[1])
    reversal = np.zeros_like(direction, dtype=bool)
    for t, current in enumerate(direction):
        reversal[t] = (current != 0) & (last != 0) & (current != last)
        last = np.where(current != 0, current, last)
    active = np.flatnonzero(np.asarray(wrapper.delta.joint_scale) != 0)
    transmission = transmission_arrays(tracking)
    engaged = np.ones_like(v, dtype=bool)
    for j, spec in enumerate(wrapper.shadow_plan):
        engaged[:, wrapper.index.joint_names.index(spec.parent_joint)] = transmission["transmitting"][:, j]
    eef_speed = np.linalg.norm(np.diff(tracking.eef_pos_cmd, axis=0,
                            prepend=tracking.eef_pos_cmd[:1]), axis=1) * CONTROL_FREQ
    ref_speed = np.linalg.norm(np.diff(reference.positions, axis=0,
                            prepend=reference.positions[:1]), axis=1) * CONTROL_FREQ
    action_derivative = np.diff(tracking.actions[:, :6], axis=0,
                               prepend=tracking.actions[:1, :6]) * CONTROL_FREQ
    arrays = {
        "joint_position_rad": q, "joint_velocity_rad_s": v,
        "shadow_position_rad": shadow_q, "shadow_velocity_rad_s": shadow_v,
        "backlash_half_width_rad": np.asarray([
            float(spec.half_range) for spec in wrapper.shadow_plan
        ]),
        "joint_reversal": reversal, "engaged": engaged,
        "limit_active": transmission["limit_active"],
        "boundary_side": transmission["boundary_side"],
        "transmission_torque_nm": transmission["transmission_torque_nm"],
        "penetration_rad": transmission["penetration_rad"],
        "eef_speed_m_s": eef_speed, "reference_speed_m_s": ref_speed,
        "action_derivative_per_s": action_derivative,
        "gravity_torque_nm": tracking.tau_g,
        "active_joint_indices": active,
    }
    summary = {
        "active_joint_indices": active.tolist(),
        "joint_reversals": reversal.sum(axis=0).tolist(),
        "active_joint_reversals": int(reversal[:, active].sum()),
        "active_disengaged_frame_fraction": float((~engaged[:, active]).any(axis=1).mean()),
        "shadow_position_abs_over_half_width": (
            np.abs(shadow_q) / np.asarray([
                float(spec.half_range) for spec in wrapper.shadow_plan
            ])[None, :]
        ).max(axis=0).tolist(),
        "joint_range_rad": np.ptp(q, axis=0).tolist(),
        "boundary_side_counts": {str(side): int((transmission["boundary_side"] == side).sum()) for side in (-1, 0, 1)},
        "max_penetration_rad": float(transmission["penetration_rad"].max()),
        "max_transmission_torque_nm": float(np.abs(transmission["transmission_torque_nm"]).max()),
        "gravity_torque_range_nm": np.ptp(tracking.tau_g, axis=0).tolist(),
        "eef_speed_quantiles_m_s": np.quantile(eef_speed, [0, .25, .5, .75, 1]).tolist(),
        "reference_speed_quantiles_m_s": np.quantile(ref_speed, [0, .25, .5, .75, 1]).tolist(),
        "phase_counts": {str(int(x)): int((reference.phase == x).sum())
                         for x in np.unique(reference.phase)},
        "interpretation": "Recorded exposures; counts alone do not establish compensation leakage.",
    }
    return arrays, summary


def append_teacher_details(path, result, counterpart, wrapper, reference, reference_spec,
                           config, clean_config):
    import h5py
    import numpy as np

    tracking = result.final_tracking
    arrays, feature_summary = mechanism_features(tracking, wrapper, reference)
    with h5py.File(path, "r+") as handle:
        demo = handle["data/demo_0"]
        for name, value in {
            "success_trace": tracking.success_trace,
            "actions_unclipped": tracking.actions_raw,
            "eef_rot_cmd": tracking.eef_rot_cmd,
            "eef_rot_ach": tracking.eef_rot_ach,
            "mode_codes": tracking.mode_codes,
        }.items():
            demo.create_dataset(name, data=value, compression="gzip")
        demo.attrs["terminal_success"] = bool(tracking.success)
        demo.attrs["peak_cfrc_ext"] = float(tracking.peak_cfrc_ext)
        demo.attrs["defective_teacher_config_json"] = json.dumps(config_payload(config), sort_keys=True)
        demo.attrs["clean_teacher_config_json"] = json.dumps(config_payload(clean_config), sort_keys=True)
        ref = demo.create_group("reference_trace")
        for name in ("positions", "rotations", "gripper", "phase", "waypoint_index"):
            ref.create_dataset(name, data=getattr(reference, name), compression="gzip")
        ref.attrs["waypoint_names_json"] = json.dumps(reference.waypoint_names)
        ref.attrs["spec_json"] = json.dumps(reference_spec.to_json(), sort_keys=True)
        features = demo.create_group("mechanism_features")
        for name, value in arrays.items():
            features.create_dataset(name, data=value, compression="gzip")
        hist = demo.create_group("ilc_history")
        hist.create_dataset("feedforward", data=np.stack([x.feedforward for x in result.snapshots]),
                            compression="gzip")
        for name in ("k", "e_rms", "position_rms", "orientation_rms", "mode_hash", "success",
                     "clipped_fraction", "peak_cfrc_ext"):
            hist.create_dataset(name, data=np.asarray([getattr(x, name) for x in result.snapshots]))
        hist.attrs["outcome"] = result.outcome.value
        hist.attrs["k_conv"] = result.k_conv if result.k_conv is not None else -1
        hist.attrs["counterpart_outcome"] = counterpart.outcome.value
        c = demo["c_true"][:, :6].astype(float)
        feature_summary["compensation_rms_action_units"] = float(np.sqrt(np.mean(c * c)))
        feature_summary["compensation_max_norm_action_units"] = float(np.linalg.norm(c, axis=1).max())
        demo.attrs["mechanism_summary_json"] = json.dumps(feature_summary, sort_keys=True)
    return feature_summary


def collect(args, cell, protocol):
    from course_tools.backlash_physics import PHYSICS_VERSION, STATE_CONTRACT
    from portal_trace.diagnostics import regimes
    from portal_trace.lockfile import source_commit
    from portal_trace.runners.backends import get_backend
    from portal_trace.runners.collect import (
        CollectionPlan, _paired_episode_provenance,
        _canonical_sha256,
    )
    from portal_trace.runners.pilot import pilot_thresholds
    from portal_trace.teacher import ilc, osc, query, refs
    from course_tools.limit_backlash_data import RecordingTracker, write_episode

    raw = cell / "raw" / args.task / args.family / args.defect / f"L{args.level}" / "demo.hdf5"
    plan = CollectionPlan(task=args.task, family=args.family, defect=args.defect,
        level=args.level, n_demos=1, env_seeds=(args.seed_env,), geom_variant="G0", output=raw)
    clean_raw = plan.clean_output
    if raw.exists() or clean_raw.exists():
        if not (raw.exists() and clean_raw.exists() and (cell / "collection.json").exists()):
            raise FileExistsError(f"Incomplete earlier pair in {cell}; preserve and inspect its files")
        previous = json.loads((cell / "collection.json").read_text())
        for role in ("defective", "clean"):
            row = previous[role]
            if sha256(row["raw_path"]) != row["raw_sha256"]:
                raise ValueError(f"Previously collected {role} file changed")
        return previous

    thresholds = pilot_thresholds(json.loads(args.threshold_report.read_text()))
    pair = make_pair(args.task, args.family, args.defect, args.level, args.seed_env)
    try:
        initial_physics = osc.reset_twin_pair(pair, args.seed_env)
        initial_actuation = pair.clean.actuation_state().copy()
        reference_spec = refs.load_spec(args.task)
        reference = refs.resolve(reference_spec, pair.clean.sim)
        cfg = ilc.ILCConfig.for_delta(pair.delta, k_max=args.k_max)
        clean_cfg = ilc.ILCConfig.for_delta(pair.delta.zero(), k_max=args.k_max)
        reset = lambda: osc.reset_twin_pair(pair, args.seed_env)
        defective = ilc.run_ilc(pair.defective, reference, family=args.family,
            seed_env=args.seed_env, tol=args.tol, config=cfg, reset=reset, tracker=RecordingTracker())
        clean = ilc.run_ilc(pair.clean, reference, family=args.family,
            seed_env=args.seed_env, tol=args.tol, config=clean_cfg, reset=reset, tracker=RecordingTracker())
        runtime_json = json.dumps(get_backend("cpu_mujoco").fingerprint(
            lite_physics=True).to_json(), sort_keys=True, separators=(",", ":"))
        producer = source_commit()
        physics_exports = {"defective": pair.defective.export_physics(), "clean": pair.clean.export_physics()}
        provenance = _paired_episode_provenance(plan, pair, reference_spec, reference,
            seed_env=args.seed_env, initial_physics=initial_physics,
            initial_actuation=initial_actuation, config=cfg,
            controller_sha256=defective.final_tracking.controller_config_sha256,
            clean_controller_sha256=clean.final_tracking.controller_config_sha256,
            run_source_commit=producer, runtime_json=runtime_json)
        # The helper predates separate source/clean ILC configurations.
        provenance["teacher_identity_clean_sha256"] = _canonical_sha256({
            "implementation": "portal_trace.teacher.ilc.run_ilc", **config_payload(clean_cfg),
            "source_commit": producer, "role": "clean", "delta": pair.delta.zero().to_json()})
        # Bind the new physical implementation explicitly; git HEAD alone does
        # not identify uncommitted course-only physics.
        provenance["course_physics_sha256"] = sha256(Path(__file__).with_name("backlash_physics.py"))
        attrs = {"course_physics_version": PHYSICS_VERSION,
            "course_physics_sha256": provenance["course_physics_sha256"],
            "course_collection_schema": SCHEMA,
            "course_protocol_sha256": protocol["protocol_sha256"],
            "course_collector_sha256": sha256(__file__),
            "course_source": SOURCES[args.defect], "course_split": args.split,
            "threshold_report_sha256": sha256(args.threshold_report)}
        records = {}
        for role, result, counterpart, body, path, delta in (
            ("defective", defective, clean, pair.defective, raw, pair.delta),
            ("clean", clean, defective, pair.clean, clean_raw, pair.delta.zero()),
        ):
            write_episode(path, plan=plan, pair=pair, role=role, result=result,
                clean_result=clean, reference=reference, provenance=provenance,
                export=physics_exports[role], attrs=attrs, thresholds=thresholds, regimes=regimes)
            features = append_teacher_details(path, result, counterpart, body, reference,
                                               reference_spec, cfg, clean_cfg)
            import h5py
            export = dict(physics_exports[role])
            export.pop("model_xml")
            with h5py.File(path, "r+") as handle:
                for obj in (handle, handle["data/demo_0"]):
                    obj.attrs["defect_state_schema_version"] = 3
                    obj.attrs["defect_state_semantics"] = STATE_CONTRACT
                    obj.attrs["course_physics_version"] = PHYSICS_VERSION
                    obj.attrs["physics_export_json"] = json.dumps(export, sort_keys=True)
                handle["data/demo_0"].attrs["model_file"] = physics_exports[role]["model_xml"]
            records[role] = {"raw_path": str(path), "raw_sha256": sha256(path),
                "physics_version": PHYSICS_VERSION, "physics_export": export,
                "source": "clean" if role == "clean" else SOURCES[args.defect],
                "success": bool(result.final_tracking.success), "t_success": result.final_tracking.t_success,
                "ilc_status": result.outcome.value, "ilc_iterations": result.iterations,
                "position_rms_m": float(result.final_tracking.position_rms()),
                "orientation_rms_rad": float(result.final_tracking.orientation_rms()),
                "mechanism_features": features, "pair_id": provenance["pair_id"]}
        # This version has limit-reaction semantics, not the old equality
        # contract. Publish a course pair marker after both new files exist.
        atomic_json(cell / "pair-complete.json", {"schema": SCHEMA,
            "physics_version": PHYSICS_VERSION,
            "pair_id": provenance["pair_id"],
            "raw": {role: {key: records[role][key] for key in ("raw_path", "raw_sha256")}
                    for role in ("defective", "clean")}})
        result = {"schema": SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
                  "delta": pair.delta.to_json(), "normal_reference_delta": pair.delta.zero().to_json(),
                  "teacher_configs": {"defective": config_payload(cfg), "clean": config_payload(clean_cfg)},
                  "runtime": json.loads(runtime_json),
                  "probe_adapter": {"module": "course_tools.collect_limit_backlash",
                      "factory": "make_pair", "module_sha256": sha256(__file__)}, **records}
        atomic_json(cell / "collection.json", result)
        return result
    finally:
        pair.defective.close()
        pair.clean.close()


def materialise_record(row, cell, args):
    from course_tools.materialise_physics_episode import materialise
    path = cell / "observations" / f"{row['source']}.hdf5"
    return materialise(Path(row["raw_path"]), path, level=args.level)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Stack")
    parser.add_argument("--family", default="Panda")
    parser.add_argument("--defect", choices=("D4hi",), default="D4hi")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seed-env", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--threshold-report", type=Path, required=True)
    parser.add_argument("--tol", type=float, required=True,
                        help="Task-space tolerance used by the existing ILC stopping rule, in metres")
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--materialise", action="store_true")
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args(argv)
    if not 0 <= args.seed_env < 2**32 or args.tol <= 0 or args.level < 0:
        parser.error("seed, tolerance, or level is invalid")
    os.environ["MUJOCO_GL"] = args.render_backend
    if args.render_backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.setdefault("LP_NUM_THREADS", "1")
    args.threshold_report = args.threshold_report.resolve()
    cell = (args.output_root.resolve() / args.task / args.family /
            f"{args.defect}-L{args.level}" / f"seed-{args.seed_env}")
    cell.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": SCHEMA, "task": args.task, "family": args.family,
        "defect": args.defect, "source": SOURCES[args.defect], "level": args.level,
        "seed_env": args.seed_env, "split": args.split, "tol_m": args.tol, "k_max": args.k_max,
        "threshold_report": str(args.threshold_report),
        "threshold_report_sha256": sha256(args.threshold_report),
        "collector_sha256": sha256(__file__),
        "physics_sha256": sha256(Path(__file__).with_name("backlash_physics.py")),
        "data_writer_sha256": sha256(Path(__file__).with_name("limit_backlash_data.py")),
        "physics_version": "dl_course.backlash.unilateral-limit-v1",
        "control_hz": 20, "horizon": 400,
        "clean_teacher": "normal-reference ILC; same transmission coordinates with vanishing reference gap",
        "controller": "portal_trace.teacher.osc.FROZEN_GAINS",
        "task_initialization": "paired clean-settled initial physics and gripper accumulator",
        "teacher_selection": "last ILC iteration under the fixed stopping rule; failures retained"}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = cell / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Collection protocol differs from existing {protocol_path}")
    else:
        atomic_json(protocol_path, protocol)
    start = time.monotonic()
    records = collect(args, cell, protocol)
    episodes = []
    for role in ("defective", "clean"):
        row = records[role]
        materialised = materialise_record(row, cell, args) if args.materialise else {}
        episodes.append({**row, **materialised, "path": materialised.get("path"),
            "demo": "demo_0", "seed_env": args.seed_env, "split": args.split,
            "level": args.level, "defect": "clean" if role == "clean" else args.defect,
            "clean_of": args.defect if role == "clean" else None,
            "task": args.task, "family": args.family, "c_true_key": "c_true",
            "normal_source_duplicate_warning": role == "clean"})
    manifest = {"schema": SCHEMA, "status": "complete", "episodes": episodes,
                "protocol_sha256": protocol["protocol_sha256"],
                "collection_seconds_this_invocation": time.monotonic() - start,
                "materialised": args.materialise,
                "normal_source_note": "Each pair supplies its own reference. Select one canonical normal source per scene; do not count repeated clean twins as independent examples."}
    atomic_json(cell / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "manifest": str(cell / "manifest.json"),
                      "episodes": [{"source": r["source"], "success": r["success"],
                                    "ilc_status": r["ilc_status"]} for r in episodes]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
