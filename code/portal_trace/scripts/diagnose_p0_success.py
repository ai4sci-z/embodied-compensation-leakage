"""Inspect the frame-level success history behind one P0 paired rollout.

Formal P0 shards intentionally contain only the binary outcome and first
success frame.  This engineering diagnostic runs the tracker while retaining
the per-frame success predicate; it can use either the uncompensated path or
the final ``u_inf`` stored by an ILC pilot.  Its output is diagnostic evidence
only and is not accepted by the formal P0 assembler.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.configs.loader import load_bundle
from portal_trace.defects.delta import Delta, default_joint_scale
from portal_trace.defects.inject import build_twin_pair
from portal_trace.lockfile import source_commit
from portal_trace.registry.families import descriptor
from portal_trace.registry.tasks import HORIZON, task as task_spec
from portal_trace.teacher import osc, refs
from portal_trace.types import SeedTriple

__all__ = ["run_diagnostic", "summarize_success_trace", "main"]

_UNUSED_TRAIN_SEED = -1


def _worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def _true_runs(trace: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate(np.append(trace, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append({"start": start, "stop": index, "length": index - start})
            start = None
    return runs


def summarize_success_trace(trace: Sequence[bool]) -> dict[str, Any]:
    values = np.asarray(trace, dtype=bool)
    if values.shape != (HORIZON,):
        raise ValueError(f"success trace must have {HORIZON} frames, got {values.shape}")
    runs = _true_runs(values)
    terminal_run = runs[-1]["length"] if runs and runs[-1]["stop"] == HORIZON else 0
    successful = np.flatnonzero(values)
    return {
        "ever_success": bool(successful.size),
        "first_success": int(successful[0]) if successful.size else -1,
        "last_success": int(successful[-1]) if successful.size else -1,
        "success_frames": int(values.sum()),
        "success_fraction": float(values.mean()),
        "terminal_success": bool(values[-1]),
        "terminal_run_length": int(terminal_run),
        "max_run_length": max((run["length"] for run in runs), default=0),
        "last_5_all_success": bool(values[-5:].all()),
        "last_20_all_success": bool(values[-20:].all()),
        "runs": runs,
    }


def _task_state_snapshot(env: Any) -> dict[str, Any]:
    """Capture task-specific predicate inputs at the end of a diagnostic run.

    The formal shard intentionally stores only the endpoint.  Diagnostics need
    the predicate inputs as well: a zero success rate is actionable only when we
    can tell whether the reference missed the object, the object missed the
    task geometry, or the environment's composite predicate failed one clause.
    This function is read-only and deliberately remains outside the formal
    result schema.
    """
    out: dict[str, Any] = {"env_class": type(env).__name__}
    if hasattr(env, "hinge_qpos_addr"):
        out["hinge_qpos_rad"] = float(env.sim.data.qpos[env.hinge_qpos_addr])
        if hasattr(env, "handle_qpos_addr"):
            out["handle_qpos_rad"] = float(env.sim.data.qpos[env.handle_qpos_addr])
        handle = np.asarray(env.sim.data.site_xpos[env.door_handle_site_id], dtype=np.float64)
        distances = [
            np.linalg.norm(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]] - handle)
            for arm in env.robots[0].arms
        ]
        out["handle_position_m"] = [float(x) for x in handle]
        out["min_eef_handle_distance_m"] = float(min(distances))
        finger_geoms = [
            geom
            for arm in env.robots[0].arms
            for group in env.robots[0].gripper[arm].important_geoms.values()
            for geom in group
        ]
        out["finger_handle_contact"] = bool(env.check_contact(finger_geoms, ["Door_handle"]))
    if hasattr(env, "wiped_markers"):
        out["wiped_markers"] = int(len(env.wiped_markers))
        out["num_markers"] = int(env.num_markers)
        out["arm_collision_or_joint_limit_steps"] = int(env.collisions)
        out["excess_force_steps"] = int(env.f_excess)
        if hasattr(env, "model") and hasattr(env.model, "mujoco_arena"):
            wiped = {getattr(m, "name", str(m)) for m in env.wiped_markers}
            out["unwiped_markers"] = [
                getattr(m, "name", str(m))
                for m in env.model.mujoco_arena.markers
                if getattr(m, "name", str(m)) not in wiped
            ]
    if hasattr(env, "_check_frame_assembled"):
        out["frame_assembled"] = bool(env._check_frame_assembled())
    if hasattr(env, "_check_tool_on_frame"):
        out["tool_on_frame"] = bool(env._check_tool_on_frame())
    if hasattr(env, "objects_in_bins"):
        out["objects_in_bins"] = [int(v) for v in env.objects_in_bins]
        object_rows: list[dict[str, Any]] = []
        for i, obj in enumerate(env.objects):
            body = env.sim.data.body_xpos[env.obj_body_id[obj.name]]
            dists = [
                np.linalg.norm(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]] - body)
                for arm in env.robots[0].arms
            ]
            object_rows.append({
                "name": obj.name,
                "position": [float(x) for x in body],
                "not_in_bin": bool(env.not_in_bin(body, i)),
                "min_eef_distance_m": float(min(dists)),
                "r_reach": float(1.0 - np.tanh(10.0 * min(dists))),
            })
        out["objects"] = object_rows
    return out


def _task_trace_summary(
    env: Any,
    result: osc.TrackingResult,
    reference: refs.ReferenceTrajectory,
) -> dict[str, Any]:
    """Replay recorded states to locate where a task predicate stopped progressing."""
    tight_state = getattr(env, "tight_fit_state", None)
    if not (
        hasattr(env, "hinge_qpos_addr")
        or hasattr(env, "_check_frame_assembled")
        or callable(tight_state)
    ):
        return {}
    saved = osc.physics_state(env.sim)
    # DefectWrapper delegates attribute reads but not writes.  The stateful
    # entry datum therefore has to be cleared on the bound environment itself;
    # assigning it on the wrapper leaves the rollout's terminal datum live and
    # makes replay appear to have entered the chamfer at frame zero.
    tight_owner = getattr(tight_state, "__self__", env)
    saved_tight_entry = getattr(tight_owner, "_entry_underside_z", None)

    def active_contact_pairs() -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        model = env.sim.model
        data = env.sim.data
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom_a = model.geom_id2name(int(contact.geom1))
            geom_b = model.geom_id2name(int(contact.geom2))
            pairs.append({
                "geom1": geom_a,
                "geom2": geom_b,
                "distance_m": float(contact.dist),
                "geom1_position_m": np.asarray(
                    data.geom_xpos[int(contact.geom1)], dtype=np.float64
                ).copy().tolist(),
                "geom2_position_m": np.asarray(
                    data.geom_xpos[int(contact.geom2)], dtype=np.float64
                ).copy().tolist(),
            })
        return pairs
    try:
        if callable(tight_state):
            # tight_fit_state() establishes the first chamfer-entry datum. Replay
            # from a cleared datum so the diagnostic follows the same stateful
            # predicate sequence as the original rollout.
            tight_owner._entry_underside_z = None
            rows: list[dict[str, Any]] = []
            samples: dict[str, dict[str, Any]] = {}
            sample_frames = {59, 103, 123, 163, 231, 279, 335, 355, 399}
            fit = env.tight_fit_spec
            peg_id = env.sim.model.body_name2id(fit.peg_body)
            nut = env.nuts[fit.peg_index]
            nut_id = env.obj_body_id[nut.name]
            handle_id = env.sim.model.site_name2id(f"{nut.name}_handle_site")
            available_geom_names = set(env.sim.model.geom_names)
            robot_grasp_geoms = [
                geom
                for arm in env.robots[0].arms
                for group in env.robots[0].gripper[arm].important_geoms.values()
                for geom in group
            ]
            nut_contact_geoms = list(nut.contact_geoms)
            for state_index, state in enumerate(result.states):
                osc.set_physics_state(env.sim, state)
                env.sim.forward()
                row = dict(tight_state())
                # MuJoCo exposes these arrays as views into one mutable mjData
                # buffer.  Copy each frame or every row aliases the final replay
                # state and all object-displacement diagnostics collapse to zero.
                nut_position = np.asarray(
                    env.sim.data.body_xpos[nut_id], dtype=np.float64
                ).copy()
                handle_position = np.asarray(
                    env.sim.data.site_xpos[handle_id], dtype=np.float64
                ).copy()
                eef_position = np.asarray(
                    env.sim.data.site_xpos[
                        env.robots[0].eef_site_id[env.robots[0].arms[0]]
                    ],
                    dtype=np.float64,
                ).copy()
                waypoint = reference.waypoint_names[int(reference.waypoint_index[state_index])]
                row.update({
                    "waypoint": waypoint,
                    "robot_nut_contact": bool(
                        env.check_contact(robot_grasp_geoms, nut_contact_geoms)
                    ),
                    "nut_position_m": nut_position,
                    "handle_position_m": handle_position,
                    "eef_position_m": eef_position,
                    "eef_handle_distance_m": float(np.linalg.norm(eef_position - handle_position)),
                    "eef_nut_distance_m": float(np.linalg.norm(eef_position - nut_position)),
                    "gripper_action": float(result.actions[state_index, 6]),
                    "gripper_actuation": result.actuation[state_index].tolist(),
                })
                rows.append(row)
                if state_index in sample_frames:
                    samples[str(state_index)] = {
                        **{
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in row.items()
                        },
                        "peg_position_m": env.sim.data.body_xpos[peg_id].tolist(),
                        "command_position_m": result.eef_pos_cmd[state_index].tolist(),
                        "contact_pairs": active_contact_pairs(),
                        "robot_grasp_geom_positions_m": {
                            name: env.sim.data.geom_xpos[
                                env.sim.model.geom_name2id(name)
                            ].tolist()
                            for name in robot_grasp_geoms
                            if name in available_geom_names
                        },
                    }

            radial = np.asarray([float(row["radial_m"]) for row in rows])
            entered = [i for i, row in enumerate(rows) if row["entry_underside_z"] is not None]
            descended = [
                (i, float(row["descent_m"]))
                for i, row in enumerate(rows)
                if row["descent_m"] is not None
            ]
            all_clauses = [
                i
                for i, row in enumerate(rows)
                if row["radial_ok"] and row["descent_ok"] and row["released_ok"]
            ]
            maximum_descent = max(descended, key=lambda item: item[1]) if descended else None
            closest = int(radial.argmin())
            contact_frames = np.flatnonzero([row["robot_nut_contact"] for row in rows])
            initial_nut = rows[0]["nut_position_m"]
            nut_displacement = np.asarray([
                np.linalg.norm(row["nut_position_m"] - initial_nut) for row in rows
            ])
            waypoint_metrics: dict[str, dict[str, Any]] = {}
            for waypoint in dict.fromkeys(row["waypoint"] for row in rows):
                ids = [i for i, row in enumerate(rows) if row["waypoint"] == waypoint]
                waypoint_metrics[waypoint] = {
                    "start_frame": ids[0],
                    "stop_frame": ids[-1] + 1,
                    "robot_nut_contact_frames": int(sum(
                        rows[i]["robot_nut_contact"] for i in ids
                    )),
                    "minimum_eef_handle_distance_m": float(min(
                        rows[i]["eef_handle_distance_m"] for i in ids
                    )),
                    "maximum_nut_displacement_m": float(nut_displacement[ids].max()),
                    "terminal_nut_position_m": rows[ids[-1]]["nut_position_m"].tolist(),
                    "terminal_eef_position_m": rows[ids[-1]]["eef_position_m"].tolist(),
                    "terminal_gripper_actuation": rows[ids[-1]]["gripper_actuation"],
                }
            return {
                "minimum_radial_m": float(radial[closest]),
                "minimum_radial_frame": closest,
                "radial_tolerance_m": float(rows[closest]["tol_m"]),
                "first_chamfer_entry_frame": entered[0] if entered else -1,
                "maximum_descent_m": (
                    maximum_descent[1] if maximum_descent is not None else None
                ),
                "maximum_descent_frame": (
                    maximum_descent[0] if maximum_descent is not None else -1
                ),
                "first_all_clauses_frame": all_clauses[0] if all_clauses else -1,
                "all_clauses_frames": len(all_clauses),
                "first_robot_nut_contact_frame": (
                    int(contact_frames[0]) if contact_frames.size else -1
                ),
                "last_robot_nut_contact_frame": (
                    int(contact_frames[-1]) if contact_frames.size else -1
                ),
                "robot_nut_contact_frames": int(contact_frames.size),
                "maximum_nut_displacement_m": float(nut_displacement.max()),
                "maximum_nut_displacement_frame": int(nut_displacement.argmax()),
                "waypoint_metrics": waypoint_metrics,
                "sampled_frames": samples,
            }

        if hasattr(env, "hinge_qpos_addr"):
            hinge: list[float] = []
            latch: list[float] = []
            handle_distance: list[float] = []
            for state in result.states:
                osc.set_physics_state(env.sim, state)
                env.sim.forward()
                hinge.append(float(env.sim.data.qpos[env.hinge_qpos_addr]))
                if hasattr(env, "handle_qpos_addr"):
                    latch.append(float(env.sim.data.qpos[env.handle_qpos_addr]))
                handle = env.sim.data.site_xpos[env.door_handle_site_id]
                handle_distance.append(float(min(
                    np.linalg.norm(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]] - handle)
                    for arm in env.robots[0].arms
                )))
            h = np.asarray(hinge)
            d = np.asarray(handle_distance)
            out: dict[str, Any] = {
                "max_hinge_qpos_rad": float(h.max()),
                "max_hinge_frame": int(h.argmax()),
                "min_eef_handle_distance_m": float(d.min()),
                "min_eef_handle_distance_frame": int(d.argmin()),
            }
            if latch:
                q = np.asarray(latch)
                out.update({
                    "max_abs_handle_qpos_rad": float(np.abs(q).max()),
                    "max_abs_handle_qpos_frame": int(np.abs(q).argmax()),
                })
            return out

        frame = []
        tool = []
        tool_metrics: list[dict[str, Any]] = []
        if hasattr(env, "_check_tool_on_frame"):
            # Keep the native predicate intact, but expose its five clauses so a
            # failed engineering rollout points to one geometric cause instead
            # of triggering another blind reference-parameter sweep.
            robot_grasp_geoms = [
                env.robots[0].gripper[arm].important_geoms[key]
                for arm in env.robots[0].gripper
                for key in env.robots[0].gripper[arm].important_geoms
            ]
            hole_geoms = [
                f"tool_hole1_hc_{i}" for i in range(env.tool_args["ngeoms"])
            ]
            hook_geom = "frame_horizontal_frame"
            hook_id = env.obj_site_id["frame_hang_site"]
            intersection_id = env.obj_site_id["frame_intersection_site"]
            hole_id = env.obj_site_id["tool_hole1_center"]
            hole_radius = env.tool_args["inner_radius_1"]
            hook_half_width = env.frame_args["frame_thickness"] / 2.0
            opposite = env.tool_args["ngeoms"] // 2
        scene_snapshots: dict[str, dict[str, Any]] = {}
        snapshot_frames = {199, 211, 343, 344, 350, 360, 370, 378, 391, 399}
        for state_index, state in enumerate(result.states):
            osc.set_physics_state(env.sim, state)
            env.sim.forward()
            frame.append(bool(env._check_frame_assembled()))
            tool.append(bool(env._check_tool_on_frame()))
            if hasattr(env, "_check_tool_on_frame"):
                hook = env.sim.data.site_xpos[hook_id]
                hook_vec = env.sim.data.site_xpos[intersection_id] - hook
                hook_length = float(np.linalg.norm(hook_vec))
                hook_vec = hook_vec / hook_length
                hole = env.sim.data.site_xpos[hole_id]
                hole_vec = hole - hook
                along = float(np.dot(hole_vec, hook_vec) / hook_length)
                ortho = hole_vec - np.dot(hole_vec, hook_vec) * hook_vec
                g1 = env.sim.data.geom_xpos[env.obj_geom_id[f"tool_hole1_hc_0"]]
                g2 = env.sim.data.geom_xpos[env.obj_geom_id[f"tool_hole1_hc_{opposite}"]]
                cross1 = np.cross(g1 - hook, hook_vec)
                cross2 = np.cross(g2 - hook, hook_vec)
                robot_contact = any(
                    env.check_contact(group, env.tool.contact_geoms)
                    for group in robot_grasp_geoms
                )
                tool_metrics.append({
                    "robot_tool_contact": bool(robot_contact),
                    "frame_tool_hole_contact": bool(
                        env.check_contact(hole_geoms, hook_geom)
                    ),
                    "line_distance_m": float(np.linalg.norm(ortho)),
                    "line_distance_limit_m": float(hole_radius - hook_half_width),
                    "opposite_sides": bool(np.dot(cross1, cross2) < 0),
                    "insertion_fraction": along,
                    "hook_position_m": hook.tolist(),
                    "hole_position_m": hole.tolist(),
                    "hole_geom0_m": g1.tolist(),
                    "hole_geom_opposite_m": g2.tolist(),
                    "hook_direction": hook_vec.tolist(),
                })
                if state_index in snapshot_frames:
                    site_names = (
                        "frame_hang_site", "frame_intersection_site", "frame_tip_site",
                        "tool_grip_default_site", "tool_hole1_center",
                    )
                    geom_names = (
                        "stand_base", "stand_wall0", "stand_wall2",
                        "frame_horizontal_frame",
                    )
                    scene_snapshots[str(state_index)] = {
                        "sites": {
                            name: env.sim.data.site_xpos[env.obj_site_id[name]].tolist()
                            for name in site_names if name in env.obj_site_id
                        },
                        "geoms": {
                            name: env.sim.data.geom_xpos[env.obj_geom_id[name]].tolist()
                            for name in geom_names if name in env.obj_geom_id
                        },
                    }
        frame_ids = np.flatnonzero(frame)
        tool_ids = np.flatnonzero(tool)
        out = {
            "frame_assembled_frames": int(len(frame_ids)),
            "first_frame_assembled": int(frame_ids[0]) if len(frame_ids) else -1,
            "last_frame_assembled": int(frame_ids[-1]) if len(frame_ids) else -1,
            "tool_on_frame_frames": int(len(tool_ids)),
            "first_tool_on_frame": int(tool_ids[0]) if len(tool_ids) else -1,
            "last_tool_on_frame": int(tool_ids[-1]) if len(tool_ids) else -1,
        }
        if tool_metrics:
            out["tool_metrics"] = {
                "minimum_line_distance_m": float(min(m["line_distance_m"] for m in tool_metrics)),
                "minimum_line_distance_frame": int(np.argmin([m["line_distance_m"] for m in tool_metrics])),
                "maximum_insertion_fraction": float(max(m["insertion_fraction"] for m in tool_metrics)),
                "maximum_insertion_frame": int(np.argmax([m["insertion_fraction"] for m in tool_metrics])),
                "any_frame_tool_hole_contact": any(m["frame_tool_hole_contact"] for m in tool_metrics),
                "any_opposite_sides": any(m["opposite_sides"] for m in tool_metrics),
                "minimum_line_state": tool_metrics[int(np.argmin([m["line_distance_m"] for m in tool_metrics]))],
                "scene_snapshots": scene_snapshots,
                "last_frames": tool_metrics[-10:],
            }
        return out
    finally:
        if callable(tight_state):
            tight_owner._entry_underside_z = saved_tight_entry
        osc.set_physics_state(env.sim, saved)
        env.sim.forward()


def _track_with_success_trace(
    env: Any,
    reference: refs.ReferenceTrajectory,
    *,
    family: str,
    seed_env: int,
    feedforward: np.ndarray | None = None,
) -> tuple[osc.TrackingResult, np.ndarray]:
    result = osc.OSCTracker().track(
        env,
        reference,
        family=family,
        seed_env=seed_env,
        feedforward=feedforward,
    )
    values = np.asarray(result.success_trace, dtype=bool)
    if len(values) != result.horizon:
        raise RuntimeError(
            f"tracker made {len(values)} success checks for {result.horizon} frames"
        )
    return result, values


def _pilot_feedforward(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as dataset:
        try:
            values = np.asarray(dataset["data/demo_0/u_inf"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"{path}: pilot dataset has no data/demo_0/u_inf") from exc
    expected = (HORIZON, 6)
    if values.shape != expected:
        raise ValueError(f"{path}: u_inf has shape {values.shape}, expected {expected}")
    return values


def run_diagnostic(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    demo_index: int,
    defective_feedforward: np.ndarray | None = None,
    clean_feedforward: np.ndarray | None = None,
) -> dict[str, Any]:
    bundle = load_bundle()
    seed_env = bundle.seeds.env_seed(task=task, family=family, demo_index=demo_index)
    eval_seed = bundle.seeds.eval_seed(task=task, family=family)
    n_dof = descriptor(family).dof
    delta = Delta.from_level(
        defect,
        level,
        joint_scale=default_joint_scale(defect, n_dof),
    )
    seeds = SeedTriple(env=seed_env, train=_UNUSED_TRAIN_SEED, eval=eval_seed)
    pair = build_twin_pair(task_spec(task), family, delta, seeds)
    try:
        osc.reset_twin_pair(pair, seed_env)
        reference = refs.resolve(refs.load_spec(task), pair.defective.sim)
        defective, defective_trace = _track_with_success_trace(
            pair.defective,
            reference,
            family=family,
            seed_env=seed_env,
            feedforward=defective_feedforward,
        )
        clean, clean_trace = _track_with_success_trace(
            pair.clean,
            reference,
            family=family,
            seed_env=seed_env,
            feedforward=clean_feedforward,
        )
        sides: dict[str, Any] = {}
        for name, env, result, trace in (
            ("defective", pair.defective, defective, defective_trace),
            ("clean", pair.clean, clean, clean_trace),
        ):
            summary = summarize_success_trace(trace)
            summary["tracker_success"] = bool(result.success)
            summary["tracker_t_success"] = int(result.t_success)
            summary["tracking"] = {
                "max_position_error_m": float(np.max(np.linalg.norm(result.error[:, :3], axis=1))),
                "terminal_position_error_m": float(np.linalg.norm(result.error[-1, :3])),
                "terminal_eef_position_m": [float(x) for x in result.eef_pos_ach[-1]],
                "terminal_command_position_m": [float(x) for x in result.eef_pos_cmd[-1]],
                "terminal_reference_position_m": [float(x) for x in reference.positions[-1]],
                "terminal_waypoint": reference.waypoint_names[int(reference.waypoint_index[-1])],
            }
            task_trace = _task_trace_summary(env, result, reference)
            if task_trace:
                summary["predicate_trace"] = task_trace
            tight_state = getattr(env, "tight_fit_state", None)
            if callable(tight_state):
                summary["final_task_state"] = tight_state()
            summary["predicate_inputs"] = _task_state_snapshot(env)
            sides[name] = summary
    finally:
        pair.defective.close()
        pair.clean.close()

    return {
        "schema_version": 1,
        "producer": "portal_trace.scripts.diagnose_p0_success",
        "status": "engineering_diagnostic_not_formal_result",
        "source_commit": source_commit(),
        "worktree_dirty": _worktree_dirty(),
        "task": task,
        "family": family,
        "defect": defect,
        "level": level,
        "demo_index": demo_index,
        "seed_env": seed_env,
        "feedforward": "pilot_u_inf" if defective_feedforward is not None else "none",
        "sides": sides,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--defect", required=True)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--demo-index", required=True, type=int)
    parser.add_argument("--defective-pilot", type=Path)
    parser.add_argument("--clean-pilot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if (args.defective_pilot is None) != (args.clean_pilot is None):
        parser.error("--defective-pilot and --clean-pilot must be supplied together")
    defective_feedforward = (
        _pilot_feedforward(args.defective_pilot)
        if args.defective_pilot is not None else None
    )
    clean_feedforward = (
        _pilot_feedforward(args.clean_pilot)
        if args.clean_pilot is not None else None
    )

    report = run_diagnostic(
        task=args.task,
        family=args.family,
        defect=args.defect,
        level=args.level,
        demo_index=args.demo_index,
        defective_feedforward=defective_feedforward,
        clean_feedforward=clean_feedforward,
    )
    if args.defective_pilot is not None:
        report["pilot_sources"] = {
            "defective": str(args.defective_pilot),
            "clean": str(args.clean_pilot),
        }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
