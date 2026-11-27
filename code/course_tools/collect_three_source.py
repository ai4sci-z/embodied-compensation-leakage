#!/usr/bin/env python3
"""Collect a paired teacher episode for delay, backlash, or gravity droop.

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
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {"D2": "delay", "D4hi": "backlash", "D4lo": "backlash", "D3b": "droop"}
SCHEMA = "dl_course.three_source_collection/1"


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
    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import DefectWrapper, TwinPair
    from portal_trace.registry.tasks import task as task_spec
    from portal_trace.teacher import osc
    from portal_trace.types import SeedTriple

    spec = task_spec(task)
    if spec.tight_variant_of is not None:
        from portal_trace.registry.tight import ensure_registered
        ensure_registered(spec)
    env = osc.make_env(task, family=family, seed_env=seed_env)
    bodies = []
    try:
        n_joints = len(env.robots[0].robot_joints)
        delta = Delta.from_level(defect, level,
            joint_scale=default_joint_scale(defect, n_joints))
        seeds = SeedTriple(env=seed_env, train=-1, eval=0)
        bodies.append(DefectWrapper(env, delta, seeds=seeds, family=family))
        env = osc.make_env(task, family=family, seed_env=seed_env)
        bodies.append(DefectWrapper(env, delta.zero(), seeds=seeds, family=family))
        if (bodies[0].sim.model.nq, bodies[0].sim.model.nv) != (
                bodies[1].sim.model.nq, bodies[1].sim.model.nv):
            raise ValueError("The two members of this pair have different state dimensions")
        return TwinPair(defective=bodies[0], clean=bodies[1], delta=delta,
                        family=family, task=task,
                        extra_nodes=bodies[0].index.shadow_names)
    except BaseException:
        for body in bodies:
            body.close()
        if not bodies or env is not bodies[-1].env:
            env.close()
        raise


def mechanism_features(tracking, wrapper, reference):
    import numpy as np
    from portal_trace.defects.inject import engagement_flags, ZERO_VELOCITY_BAND
    from portal_trace.registry.tasks import CONTROL_FREQ

    nq = int(wrapper.sim.model.nq)
    q = tracking.states[:, np.asarray(wrapper.index.qpos_index)]
    v = tracking.states[:, nq + np.asarray(wrapper.index.qvel_index)]
    direction = np.where(np.abs(v) > ZERO_VELOCITY_BAND, np.sign(v), 0)
    last = np.zeros(v.shape[1])
    reversal = np.zeros_like(direction, dtype=bool)
    for t, current in enumerate(direction):
        reversal[t] = (current != 0) & (last != 0) & (current != last)
        last = np.where(current != 0, current, last)
    active = np.flatnonzero(np.asarray(wrapper.delta.joint_scale) != 0)
    engaged = engagement_flags(tracking.defect_state, v.shape[1])
    eef_speed = np.linalg.norm(np.diff(tracking.eef_pos_cmd, axis=0,
                            prepend=tracking.eef_pos_cmd[:1]), axis=1) * CONTROL_FREQ
    ref_speed = np.linalg.norm(np.diff(reference.positions, axis=0,
                            prepend=reference.positions[:1]), axis=1) * CONTROL_FREQ
    action_derivative = np.diff(tracking.actions[:, :6], axis=0,
                               prepend=tracking.actions[:1, :6]) * CONTROL_FREQ
    arrays = {
        "joint_position_rad": q, "joint_velocity_rad_s": v,
        "joint_reversal": reversal, "engaged": engaged,
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
        "joint_range_rad": np.ptp(q, axis=0).tolist(),
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
    from portal_trace.diagnostics import regimes
    from portal_trace.lockfile import source_commit
    from portal_trace.runners.backends import get_backend
    from portal_trace.runners.collect import (
        CollectionPlan, _clean_episode_from_ilc, _episode_from_ilc,
        _paired_episode_provenance, _publish_pair_marker, _write_hdf5,
        _canonical_sha256,
    )
    from portal_trace.runners.pilot import pilot_thresholds
    from portal_trace.teacher import ilc, osc, query, refs
    from portal_trace.types import SeedTriple

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
            seed_env=args.seed_env, tol=args.tol, config=cfg, reset=reset)
        clean = ilc.run_ilc(pair.clean, reference, family=args.family,
            seed_env=args.seed_env, tol=args.tol, config=clean_cfg, reset=reset)
        runtime_json = json.dumps(get_backend("cpu_mujoco").fingerprint(
            lite_physics=True).to_json(), sort_keys=True, separators=(",", ":"))
        producer = source_commit()
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
        seeds = SeedTriple(env=args.seed_env, train=-1, eval=0)
        defective_episode = _episode_from_ilc(plan, pair, reference, defective, clean,
            seeds=seeds, thresholds=thresholds, regimes=regimes, query=query,
            paired_provenance=provenance, allow_calibration=True)
        clean_episode = _clean_episode_from_ilc(plan, pair, reference, clean,
            seeds=seeds, thresholds=thresholds, regimes=regimes,
            paired_provenance=provenance, allow_calibration=True)
        attrs = {"course_collection_schema": SCHEMA,
            "course_protocol_sha256": protocol["protocol_sha256"],
            "course_collector_sha256": sha256(__file__),
            "course_source": SOURCES[args.defect], "course_split": args.split,
            "threshold_report_sha256": sha256(args.threshold_report)}
        ctx = SimpleNamespace(report=SimpleNamespace(backend=SimpleNamespace(value="cpu_mujoco")))
        records = {}
        for role, ep, result, counterpart, body, path, delta in (
            ("defective", defective_episode, defective, clean, pair.defective, raw, pair.delta),
            ("clean", clean_episode, clean, defective, pair.clean, clean_raw, pair.delta.zero()),
        ):
            _write_hdf5(replace(plan, output=path), [ep], delta=delta, ctx=ctx,
                        clean=role == "clean", root_attrs=attrs)
            features = append_teacher_details(path, result, counterpart, body, reference,
                                               reference_spec, cfg, clean_cfg)
            records[role] = {"raw_path": str(path), "raw_sha256": sha256(path),
                "source": "clean" if role == "clean" else SOURCES[args.defect],
                "success": bool(result.final_tracking.success), "t_success": result.final_tracking.t_success,
                "ilc_status": result.outcome.value, "ilc_iterations": result.iterations,
                "position_rms_m": float(result.final_tracking.position_rms()),
                "orientation_rms_rad": float(result.final_tracking.orientation_rms()),
                "mechanism_features": features, "pair_id": provenance["pair_id"]}
        _publish_pair_marker(plan)
        result = {"schema": SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
                  "delta": pair.delta.to_json(), "normal_reference_delta": pair.delta.zero().to_json(),
                  "teacher_configs": {"defective": config_payload(cfg), "clean": config_payload(clean_cfg)},
                  "runtime": json.loads(runtime_json), **records}
        atomic_json(cell / "collection.json", result)
        return result
    finally:
        pair.defective.close()
        pair.clean.close()


def materialise_record(row, cell, args):
    import h5py
    from portal_trace.runners.materialise import MaterialisationSpec, _materialise_one

    path = cell / "observations" / f"{row['source']}.hdf5"
    record_path = path.with_suffix(".json")
    if path.exists():
        previous = json.loads(record_path.read_text())
        if previous["raw_sha256"] != row["raw_sha256"] or previous["sha256"] != sha256(path):
            raise ValueError(f"Materialised source changed: {path}")
        return previous
    _materialise_one(MaterialisationSpec(raw=Path(row["raw_path"]), output=path, level=args.level))
    # Keep research fields alongside images; the generic converter knows only its
    # original fields, while these additions are specific to this experiment.
    with h5py.File(row["raw_path"], "r") as raw, h5py.File(path, "r+") as obs:
        src, dst = raw["data/demo_0"], obs["data/demo_0"]
        for name in ("success_trace", "actions_unclipped", "eef_rot_cmd", "eef_rot_ach",
                     "mode_codes", "reference_trace", "mechanism_features", "ilc_history"):
            if name not in dst:
                src.copy(name, dst)
    record = {"path": str(path), "sha256": sha256(path), "raw_sha256": row["raw_sha256"]}
    atomic_json(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Stack")
    parser.add_argument("--family", default="Panda")
    parser.add_argument("--defect", choices=tuple(SOURCES), required=True)
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
        "collector_sha256": sha256(__file__), "control_hz": 20, "horizon": 400,
        "clean_teacher": "normal-reference ILC configuration; lead=1 even when source delay is larger",
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
