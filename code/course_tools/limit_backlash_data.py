"""Recording and HDF5 contract for unilateral mechanical backlash episodes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from course_tools.backlash_physics import PHYSICS_VERSION, STATE_CONTRACT
from portal_trace.teacher.osc import OSCTracker, physics_state

SCHEMA = "dl_course.limit_backlash_collection/2"


class RecordingTracker(OSCTracker):
    """Capture transmission forces at the same pre-action boundary as states."""

    def track(self, env, *args, **kwargs):
        original = env.defect_state
        rows = []

        def snapshot():
            z = original()
            data = env.sim.data._data
            rows.append({
                "state": physics_state(env.sim).copy(),
                "transmissions": env.backlash_diagnostics(),
                "ctrl": data.ctrl.copy(),
                "qfrc_applied": data.qfrc_applied.copy(),
                "qacc_warmstart": data.qacc_warmstart.copy(),
            })
            return z

        env.defect_state = snapshot
        try:
            result = super().track(env, *args, **kwargs)
        finally:
            del env.defect_state
        # OSCTracker first queries the vector width, then records T frames.
        if len(rows) != result.horizon + 1:
            raise ValueError("unexpected number of pre-action transmission snapshots")
        rows = rows[1:]
        if not np.array_equal(np.stack([r["state"] for r in rows]), result.states):
            raise ValueError("transmission snapshots do not match recorded physical states")
        result.course_transmission_rows = rows
        return result


def transmission_arrays(tracking):
    rows = tracking.course_transmission_rows
    names = (
        "input_position_rad", "output_position_rad", "relative_position_rad",
        "input_velocity_rad_s", "output_velocity_rad_s", "relative_velocity_rad_s",
        "total_width_rad", "gap_to_lower_rad", "gap_to_upper_rad", "penetration_rad",
        "limit_active", "limit_rows", "boundary_side", "transmitting",
        "transmission_torque_nm", "input_actuator_torque_nm", "shadow_passive_torque_nm",
    )
    result = {key: np.asarray([[t[key] for t in r["transmissions"]] for r in rows])
              for key in names}
    for key in ("ctrl", "qfrc_applied", "qacc_warmstart"):
        result[key] = np.stack([r[key] for r in rows])
    return result


def write_episode(path, *, plan, pair, role, result, clean_result, reference,
                  provenance, export, attrs, thresholds, regimes):
    """Write the new state semantics without calling the legacy D4 writer."""
    import h5py
    from portal_trace.runners.collect import _env_args_json, _label_tracking
    from portal_trace.teacher import query

    tracking = result.final_tracking
    body = pair.clean if role == "clean" else pair.defective
    actions = np.asarray(tracking.actions, dtype=np.float32)
    if role == "clean":
        clean_actions = actions.copy()
        clipped = np.any(np.abs(tracking.actions_raw) > 1, axis=1)
    else:
        twin = query.twin_query(pair, reference, u_inf_defective=result.feedforward,
                               u_inf_clean=clean_result.feedforward)
        truth = twin.c_true(tracking.states, np.arange(tracking.horizon), tracking.defect_state)
        error = np.max(np.abs(truth.actions_defective - tracking.actions))
        if error > 1e-10:
            raise ValueError(f"recorded teacher query mismatch: {error}")
        clean_actions = truth.actions_clean.astype(np.float32)
        clipped = truth.clipped
    fields = {
        "states": tracking.states, "actions": actions, "actions_defective": actions,
        "actions_clean": clean_actions, "c_true": actions - clean_actions,
        "c_true_clipped": clipped, "rewards": tracking.rewards, "dones": tracking.dones,
        "tau_g": tracking.tau_g, "wrist_ft": tracking.wrist_ft,
        "defect_state": tracking.defect_state, "actuation_state": tracking.actuation,
        "u_inf": result.feedforward, "u_inf_clean": clean_result.feedforward,
        "phase": reference.phase, "contact": tracking.task_contact,
        "eef_pos_cmd": tracking.eef_pos_cmd, "eef_pos_ach": tracking.eef_pos_ach,
        "tracking_error": tracking.error, "joint_torque": tracking.joint_torque,
        "regime": _label_tracking(tracking, body, thresholds, regimes, allow_calibration=True),
    }
    if any(len(value) != tracking.horizon or not np.isfinite(value).all()
           for value in fields.values()):
        raise ValueError("nonfinite or misaligned recording")
    physics = {k: v for k, v in export.items() if k != "model_xml"}
    identity = {"task": plan.task, "family": plan.family,
                "defect": "clean" if role == "clean" else plan.defect,
                "level": plan.level, "clean_of": plan.defect if role == "clean" else ""}
    common = {**attrs, **identity, "course_collection_schema": SCHEMA,
              "defect_state_schema_version": 3, "defect_state_semantics": STATE_CONTRACT,
              "course_physics_version": PHYSICS_VERSION, "physics_export_json": json.dumps(physics),
              "pair_role": role, "pair_id": provenance["pair_id"],
              "wrapper_state_layout": "WrapperState.flat; engaged=nonzero unilateral limit reaction",
              "analysis_trace_semantics": "state and transmission before action; joint_torque after control step"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs.update(common)
            handle.attrs["environment"] = "cpu_mujoco"
            handle.attrs["proprio_dim"] = 70
            data = handle.create_group("data")
            data.attrs["total"] = tracking.horizon
            data.attrs["env_args"] = _env_args_json(plan)
            demo = data.create_group("demo_0")
            demo.attrs.update(common)
            demo.attrs.update(provenance)
            demo.attrs.update({"model_file": export["model_xml"], "num_samples": tracking.horizon,
                "seed_env": plan.env_seeds[0], "family_id": plan.family,
                "defect_id": identity["defect"], "defect_level": plan.level,
                "geom_variant": plan.geom_variant, "delta_json": body.delta.dumps(),
                "ilc_iters": result.iterations, "ilc_status": result.outcome.value,
                "ilc_contact_variant": result.config.contact_variant.value,
                "clean_ilc_iters": clean_result.iterations,
                "clean_ilc_status": clean_result.outcome.value, "t_success": tracking.t_success,
                "terminal_success": tracking.success,
                "controller_config_sha256": tracking.controller_config_sha256,
                "defect_impl_variant": PHYSICS_VERSION,
                "c_true_clip_saturation_share": float(np.mean(clipped)),
                "regime_role": "historical regime labels; use recorded transmission for backlash events"})
            for key, values in fields.items():
                demo.create_dataset(key, data=values, compression="gzip")
            trans = demo.create_group("transmission")
            trans.attrs["semantics"] = STATE_CONTRACT
            trans.attrs["transmissions_json"] = json.dumps(export["transmissions"])
            for key, values in transmission_arrays(tracking).items():
                trans.create_dataset(key, data=values, compression="gzip")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
