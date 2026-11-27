#!/usr/bin/env python3
"""Query recorded teacher families on a shared, explicitly defined probe input.

This evaluates the existing OSC + time-indexed ILC teachers. It does not solve
for an optimal teacher at a new history. Undefined mechanical memory is refused
rather than replaced by zeros. No trajectory integration or training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "dl_course.compensation_probes/2"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class UndefinedProbe(ValueError):
    """The saved observations do not determine this counterfactual state."""


def joint_layout(model):
    """Named joint blocks, including free objects and extra mechanical joints."""
    import mujoco
    raw = getattr(model, "_model", model)
    result = {}
    sizes = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
    for j in range(raw.njnt):
        name = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            raise UndefinedProbe("unnamed joint prevents an unambiguous state mapping")
        kind = int(raw.jnt_type[j])
        qn, vn = sizes[kind]
        result[name] = {"kind": kind, "q": int(raw.jnt_qposadr[j]),
                        "v": int(raw.jnt_dofadr[j]), "qn": qn, "vn": vn}
    return result


def map_named_states(states, source_layout, target_layout, source_nq, target_nq, target_nv):
    """Map by names, never array width; do not invent or discard internal axes."""
    missing = sorted(set(target_layout) - set(source_layout))
    extra = sorted(set(source_layout) - set(target_layout))
    if missing or extra:
        raise UndefinedProbe(f"unresolved mechanical coordinates: target_only={missing}, anchor_only={extra}")
    out = np.empty((len(states), target_nq + target_nv), dtype=float)
    for name, dst in target_layout.items():
        src = source_layout[name]
        if (src["kind"], src["qn"], src["vn"]) != (dst["kind"], dst["qn"], dst["vn"]):
            raise UndefinedProbe(f"joint type differs for {name}")
        out[:, dst["q"]:dst["q"] + dst["qn"]] = states[:, src["q"]:src["q"] + src["qn"]]
        out[:, target_nq + dst["v"]:target_nq + dst["v"] + dst["vn"]] = (
            states[:, source_nq + src["v"]:source_nq + src["v"] + src["vn"]])
    return out


def delay_history(actions, *, capacity, delay_steps, initial_buffer, initial_head, initial_effective):
    """Exact pre-step buffer recurrence under a specified shared command prefix."""
    actions = np.asarray(actions, float)
    buffer = np.asarray(initial_buffer, float).copy()
    effective = np.asarray(initial_effective, float).copy()
    if buffer.shape != (capacity, actions.shape[1]) or not 0 <= delay_steps < capacity:
        raise ValueError("delay history shape or delay is invalid")
    head = int(initial_head)
    if not 0 <= head < capacity:
        raise ValueError("initial buffer head is invalid")
    buffers, heads, effects = [], [], []
    for action in actions:
        buffers.append(buffer.copy())
        heads.append(head)
        effects.append(effective.copy())
        buffer[head] = action
        effective = buffer[(head - delay_steps) % capacity].copy()
        head = (head + 1) % capacity
    return np.asarray(buffers), np.asarray(heads), np.asarray(effects)


def load_manifest(path):
    payload = json.loads(path.read_text())
    if payload.get("schema") != "dl_course.three_source_episodes/1":
        raise ValueError("expected a three-source episode manifest")
    rows = []
    for original in payload["episodes"]:
        row = dict(original)
        for key in ("path", "raw_path"):
            p = Path(row[key])
            row[key] = str((p if p.is_absolute() else path.parent / p).resolve())
        rows.append(row)
    return payload, rows


def read_reference(group, task):
    from portal_trace.teacher.refs import ReferenceTrajectory
    ref = group["reference_trace"]
    return ReferenceTrajectory(task_id=task, **{
        name: ref[name][:] for name in ("positions", "rotations", "gripper", "phase", "waypoint_index")},
        waypoint_names=tuple(json.loads(ref.attrs["waypoint_names_json"])))


def require_same_reference(a, b):
    if a.task_id != b.task_id or a.waypoint_names != b.waypoint_names:
        raise UndefinedProbe("teachers do not follow the same task reference")
    for key in ("positions", "rotations", "gripper", "phase", "waypoint_index"):
        if not np.allclose(getattr(a, key), getattr(b, key), rtol=0, atol=1e-8):
            raise UndefinedProbe(f"reference {key} differs; matching scene seed is insufficient")


def reference_intervention(anchor, original):
    """Use one saved task reference while retaining each trained feedforward."""
    if anchor.task_id != original.task_id or anchor.waypoint_names != original.waypoint_names:
        raise UndefinedProbe("source teachers use a different task reference")
    for key in ("gripper", "phase", "waypoint_index"):
        if not np.array_equal(getattr(anchor, key), getattr(original, key)):
            raise UndefinedProbe(f"source task phase schedule differs: {key}")
    return {
        "mode": "anchor_reference_for_all_queries",
        "reference_position_max_abs_change_m": float(np.max(np.abs(anchor.positions - original.positions))),
        "reference_rotation_matrix_max_abs_change": float(np.max(np.abs(anchor.rotations - original.rotations))),
        "phase_schedule_identical": True,
        "feedforward": "preserved from each source demonstration; not reoptimized for the probe",
    }


def collection_metadata(row):
    path = Path(row["path"]).parent.parent / "collection.json"
    if not path.exists():
        raise ValueError(f"missing source collection metadata: {path}")
    meta = json.loads(path.read_text())
    return path, meta


def make_recorded_pair(row, meta):
    """An explicitly recorded adapter can supply a corrected physical model."""
    from portal_trace.teacher import osc
    adapter = row.get("probe_adapter", meta.get("probe_adapter", {}))
    module_name = adapter.get("module", "course_tools.collect_three_source")
    factory_name = adapter.get("factory", "make_pair")
    if not module_name.startswith(("course_tools.", "portal_trace.")):
        raise ValueError("probe adapters must be project modules")
    module = importlib.import_module(module_name)
    if adapter.get("module_sha256") and digest(module.__file__) != adapter["module_sha256"]:
        raise ValueError("recorded physical adapter source changed")
    defect = meta["delta"]["defect_code"]
    factory = getattr(module, factory_name)
    pair = factory(row.get("task", "Stack"), row.get("family", "Panda"),
                   defect, row["level"], row["seed_env"])
    if pair.delta.to_json() != meta["delta"]:
        close_pair(pair)
        raise ValueError("reconstructed delta differs from the recorded source")
    osc.reset_twin_pair(pair, row["seed_env"])
    return pair, adapter


def close_pair(pair):
    pair.defective.close()
    pair.clean.close()


def query_memory(wrapper, mapped, anchor_group, anchor_row, source_row, adapter, anchor_raw_sha):
    """Preserve a real mechanical history or define the exact command-prefix intervention."""
    from portal_trace.teacher import osc
    code = wrapper.delta.defect_code
    n = len(mapped)
    memory_hook = adapter.get("history_builder")
    if memory_hook:
        builder = getattr(importlib.import_module(adapter["module"]), memory_hook)
        z, explanation = builder(wrapper=wrapper, mapped_states=mapped,
                                 anchor_group=anchor_group, anchor_row=anchor_row,
                                 source_row=source_row)
        z = np.asarray(z, float)
        if z.shape != (n, wrapper.defect_state().size) or not np.isfinite(z).all():
            raise UndefinedProbe("history adapter returned invalid memory")
        return z, explanation
    if code.startswith("D4"):
        # A trace from another body is not a counterfactual engagement history.
        if anchor_raw_sha != digest(source_row["raw_path"]) or anchor_row["source"] != "backlash":
            raise UndefinedProbe("backlash internal history is not recorded for this common input; supply a physical history adapter")
        z = anchor_group["defect_state"][:]
        if z.shape != (n, wrapper.defect_state().size):
            raise UndefinedProbe("recorded backlash memory layout differs")
        return z, {"mode": "recorded_exact_source_history", "physical_rollout": True}
    if code == "D2":
        actions = anchor_group["actions"][:]
        source_initial = wrapper.state.flat().copy()
        # Frame zero is post-settling. The anchor must carry the recorded zero
        # action prefix; otherwise no initial cross-body ring state is known.
        wrapper.state.load(anchor_group["defect_state"][0])
        initial = wrapper.state
        if np.any(initial.action_buffer) or np.any(initial.effective_action) or initial.buffer_head != 0:
            raise UndefinedProbe("anchor has an unknown pre-episode command prefix")
        buffers, heads, effective = delay_history(actions[:, :wrapper.action_dim],
            capacity=initial.action_buffer.shape[0], delay_steps=int(round(wrapper._delay_steps())),
            initial_buffer=initial.action_buffer, initial_head=initial.buffer_head,
            initial_effective=initial.effective_action)
        wrapper.state.load(source_initial)
        z = []
        for t in range(n):
            osc.set_physics_state(wrapper.sim, mapped[t])
            wrapper.state.action_buffer[:] = buffers[t]
            wrapper.state.buffer_head = int(heads[t])
            wrapper.state.effective_action[:] = effective[t]
            # These clocks are read only by D10, which is not queried here.
            wrapper.state.session_time = t / 20
            z.append(wrapper.state.flat())
        return np.asarray(z), {"mode": "shared_recorded_command_prefix",
            "physical_rollout": False, "initial_buffer": "verified recorded zero prefix",
            "teacher_history_response": "OSC+ILC action does not read D2 buffer; feedforward remains source-trajectory-specific",
            "meaning": "state/command-history intervention, not a newly integrated delayed trajectory"}
    if code != "D3b" and not wrapper.delta.is_zero:
        raise UndefinedProbe(f"no declared history semantics for {code}")
    return None, {"mode": "memoryless_gravity_condition", "physical_rollout": False}


def field_geometry(fields):
    result = {}
    for name, sl in (("translation", slice(0, 3)), ("rotation", slice(3, 6))):
        labels = sorted(fields)
        matrix = np.column_stack([fields[label][:, sl].ravel() for label in labels])
        norms = np.linalg.norm(matrix, axis=0)
        active = norms > 1e-10
        normalized = matrix[:, active] / norms[active]
        singular = np.linalg.svd(normalized, compute_uv=False) if active.any() else np.array([])
        result[name] = {"labels": labels, "energy": (norms**2).tolist(),
            "active_labels": [label for label, ok in zip(labels, active) if ok],
            "normalized_gram": (normalized.T @ normalized).tolist(),
            "rank": int(np.linalg.matrix_rank(normalized)) if active.any() else 0,
            "singular_values": singular.tolist(),
            "interpretation": "geometric overlap; coefficients are not automatically independent causal contributions"}
    return result


def map_query_states(states, anchor_layout, anchor_body, target, adapter, group, anchor, source):
    hook = adapter.get("state_mapper")
    if hook:
        mapper = getattr(importlib.import_module(adapter["module"]), hook)
        mapped, meaning = mapper(states=states, anchor_wrapper=anchor_body,
            target_wrapper=target, anchor_group=group, anchor_row=anchor, source_row=source)
        mapped = np.asarray(mapped, float)
        expected = (len(states), target.sim.model.nq + target.sim.model.nv)
        if mapped.shape != expected or not np.isfinite(mapped).all():
            raise UndefinedProbe("state adapter returned invalid mapped states")
        return mapped, meaning
    target_layout = joint_layout(target.sim.model)
    extra = set(anchor_layout) - set(target_layout)
    if extra and not (set(target_layout) - set(anchor_layout)):
        return collapse_recorded_transmissions(states, anchor_layout, anchor_body,
                                               target, target_layout, group)
    mapped = map_named_states(states, anchor_layout, target_layout,
        anchor_body.sim.model.nq, target.sim.model.nq, target.sim.model.nv)
    return mapped, {"mode": "identical_named_joint_coordinates", "internal_coordinates_invented": False}


def collapse_recorded_transmissions(states, anchor_layout, anchor_body, target, target_layout, group):
    """Map a recorded two-coordinate transmission to the rigid output pose.

    The policy still receives the original anchor observation. This maps only
    the teacher query; motor encoders and force channels are not asserted equal
    across bodies. Relative history remains in the original anchor recording.
    """
    import mujoco

    raw_export = group.attrs.get("physics_export_json")
    if raw_export is None:
        raise UndefinedProbe("extra coordinates have no recorded transmission definition")
    exported = json.loads(raw_export)
    if exported.get("physics_version") != "dl_course.backlash.unilateral-limit-v1":
        raise UndefinedProbe("only the recorded unilateral backlash transmission can be collapsed")
    transmissions = exported["transmissions"]
    extra = set(anchor_layout) - set(target_layout)
    if extra != {t["shadow_name"] for t in transmissions}:
        raise UndefinedProbe("unexplained extra mechanical coordinates")
    sm = anchor_body.sim.model._model
    tm = target.sim.model._model
    reduced = np.asarray(states, float).copy()
    for t in transmissions:
        motor, shadow = t["input_name"], t["shadow_name"]
        mi = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_JOINT, motor)
        si = mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_JOINT, shadow)
        if (sm.jnt_bodyid[mi] != sm.jnt_bodyid[si]
                or not np.allclose(sm.jnt_axis[mi], sm.jnt_axis[si], rtol=0, atol=1e-12)
                or not np.allclose(sm.jnt_pos[mi], sm.jnt_pos[si], rtol=0, atol=1e-12)):
            raise UndefinedProbe("recorded transmission is not co-located and co-axial")
        a, b = anchor_layout[motor], anchor_layout[shadow]
        reduced[:, a["q"]] += reduced[:, b["q"]]
        reduced[:, sm.nq + a["v"]] += reduced[:, sm.nq + b["v"]]
    reduced_layout = {k: v for k, v in anchor_layout.items() if k not in extra}
    mapped = map_named_states(reduced, reduced_layout, target_layout, sm.nq, tm.nq, tm.nv)
    sd, td = mujoco.MjData(sm), mujoco.MjData(tm)
    sites = []
    for sid in range(sm.nsite):
        name = mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_SITE, sid)
        tid = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_SITE, name) if name else -1
        if tid >= 0:
            sites.append((sid, tid))
    if not sites:
        raise UndefinedProbe("no corresponding sites to verify the transmission collapse")
    pos_error = rot_error = 0.0
    for original, converted in zip(states, mapped):
        sd.qpos[:] = original[:sm.nq]
        td.qpos[:] = converted[:tm.nq]
        mujoco.mj_kinematics(sm, sd)
        mujoco.mj_kinematics(tm, td)
        for sid, tid in sites:
            pos_error = max(pos_error, float(np.abs(sd.site_xpos[sid] - td.site_xpos[tid]).max()))
            rot_error = max(rot_error, float(np.abs(sd.site_xmat[sid] - td.site_xmat[tid]).max()))
    if pos_error > 1e-8 or rot_error > 1e-8:
        raise UndefinedProbe(f"output-pose collapse changes sites: pos={pos_error}, rot={rot_error}")
    return mapped, {
        "mode": "recorded_coaxial_transmission_output_pose",
        "coordinate_rule": "q_output=q_motor+q_relative; v_output=v_motor+v_relative",
        "transmissions": transmissions, "site_position_max_abs_error_m": pos_error,
        "site_rotation_matrix_max_abs_error": rot_error, "checked_sites": len(sites),
        "policy_input": "original anchor RGB and proprioception history, unchanged",
        "query_estimand": "teacher action contrast at the same visible output configuration",
        "full_cross_body_observation_equality": False,
        "unequal_channels": ["motor encoder coordinates", "joint velocity", "joint acceleration", "force/torque"],
        "internal_history": "retained in anchor states; no synthetic backlash history supplied",
    }


def verify_teacher_reproduction(twin, group):
    states = group["states"][:]
    truth = twin.c_true(states, np.arange(len(states)), group["defect_state"][:])
    gaps = {"source_action": float(np.max(np.abs(truth.actions_defective - group["actions"][:]))),
            "normal_action": float(np.max(np.abs(truth.actions_clean - group["actions_clean"][:]))),
            "compensation": float(np.max(np.abs(truth.c - group["c_true"][:])))}
    if max(gaps.values()) > 1e-5:
        raise ValueError(f"reconstructed teacher does not reproduce recorded labels: {gaps}")
    return gaps


def verify_anchor_observations(group, observed, states, wrapper, layout):
    """Reject renderer/model mismatches even when copied labels look valid."""
    nq = int(wrapper.sim.model.nq)
    expected_q = states[:, np.asarray(wrapper.index.qpos_index)]
    expected_v = states[:, nq + np.asarray(wrapper.index.qvel_index)]
    checks = {}
    for name, expected in (("robot0_joint_pos", expected_q), ("robot0_joint_vel", expected_v)):
        actual = observed[f"obs/{name}"][:]
        if actual.shape != expected.shape:
            raise ValueError(f"anchor observation {name} shape differs from physical state")
        checks[name] = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
            raise ValueError(f"anchor observation {name} does not match recorded physics: max error {checks[name]}")
    xml = observed.attrs.get("model_file", group.attrs.get("model_file"))
    if xml is None:
        raise ValueError("anchor has no model XML identifying the rendered mechanical coordinates")
    if isinstance(xml, bytes):
        xml = xml.decode()
    names = {node.get("name") for node in ET.fromstring(xml).iter("joint") if node.get("name")}
    missing = sorted(set(layout) - names)
    if missing:
        raise ValueError(f"anchor model XML omits physical joints {missing}; rerender with complete injected XML")
    checks["model_xml_contains_all_physical_joints"] = True
    return checks


def build_one(anchor, sources, output, protocol, history):
    import h5py
    from portal_trace.teacher import query

    stem = f"{anchor['source']}-L{anchor['level']}-seed-{anchor['seed_env']}"
    report_path, data_path = output / f"{stem}.json", output / f"{stem}.hdf5"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["protocol_sha256"] != protocol["protocol_sha256"] or digest(data_path) != previous["sha256"]:
            raise ValueError("existing probes differ from the requested protocol")
        return previous
    _, anchor_meta = collection_metadata(anchor)
    anchor_pair, _ = make_recorded_pair(anchor, anchor_meta)
    anchor_body = anchor_pair.clean if anchor["source"] == "clean" else anchor_pair.defective
    try:
        anchor_sha = digest(anchor["raw_path"])
        if anchor.get("raw_sha256") and anchor["raw_sha256"] != anchor_sha:
            raise ValueError("anchor raw hash differs")
        if anchor.get("sha256") and digest(anchor["path"]) != anchor["sha256"]:
            raise ValueError("anchor observation hash differs")
        with h5py.File(anchor["raw_path"], "r") as anchor_file, h5py.File(anchor["path"], "r") as pixels:
            group = anchor_file[f"data/{anchor['demo']}"]
            observed = pixels[f"data/{anchor['demo']}"]
            states = group["states"][:]
            if states.shape[1] != anchor_body.sim.model.nq + anchor_body.sim.model.nv:
                raise ValueError("reconstructed anchor physics has a different state layout")
            if not np.allclose(observed["actions"][:], group["actions"][:], rtol=0, atol=1e-7):
                raise ValueError("rendered and raw anchor actions differ")
            reference = read_reference(group, anchor.get("task", "Stack"))
            layout = joint_layout(anchor_body.sim.model)
            observation_checks = verify_anchor_observations(group, observed, states, anchor_body, layout)
            steps = np.arange(len(states))
            records, fields = [], {}
            temporary = data_path.with_suffix(f".{os.getpid()}.tmp.hdf5")
            with h5py.File(temporary, "w") as result:
                result.attrs["schema"] = SCHEMA
                result.attrs["protocol_sha256"] = protocol["protocol_sha256"]
                result.attrs["anchor_metadata_json"] = json.dumps(anchor, sort_keys=True)
                result.attrs["history_frames"] = history
                # One original observation sequence is used for every source.
                observed.copy("obs", result)
                for key in ("states", "actions", "actions_clean", "c_true", "defect_state", "actuation_state", "reference_trace"):
                    group.copy(key, result)
                result.attrs["normal_policy_reference"] = "anchor actions_clean; per-source normal actions are stored separately"
                result.create_dataset("control_step", data=steps)
                result.create_dataset("history_indices", data=np.maximum(steps[:, None] - np.arange(history - 1, -1, -1), 0))
                result.attrs["joint_layout_json"] = json.dumps(layout, sort_keys=True)
                for source in sources:
                    label = f"{source['source']}-L{source['level']}"
                    pair = None
                    record = {"label": label, "source_raw_sha256": digest(source["raw_path"])}
                    if source.get("raw_sha256") and source["raw_sha256"] != record["source_raw_sha256"]:
                        raise ValueError("source raw hash differs")
                    try:
                        _, meta = collection_metadata(source)
                        pair, adapter = make_recorded_pair(source, meta)
                        target = pair.defective
                        mapped, mapping = map_query_states(states, layout, anchor_body,
                            target, adapter, group, anchor, source)
                        with h5py.File(source["raw_path"], "r") as handle:
                            sg = handle[f"data/{source['demo']}"]
                            src_ref = read_reference(sg, source.get("task", "Stack"))
                            reference_mapping = reference_intervention(reference, src_ref)
                            recorded_twin = query.twin_query(pair, src_ref,
                                u_inf_defective=sg["u_inf"][:], u_inf_clean=sg["u_inf_clean"][:])
                            reproduction = verify_teacher_reproduction(recorded_twin, sg)
                            z, semantics = query_memory(target, mapped, group, anchor, source, adapter, anchor_sha)
                            twin = query.twin_query(pair, reference,
                                u_inf_defective=sg["u_inf"][:], u_inf_clean=sg["u_inf_clean"][:])
                            truth = twin.c_true(mapped, steps, z)
                            original_reference_truth = recorded_twin.c_true(mapped, steps, z)
                            reference_mapping["source_action_max_abs_change"] = float(np.abs(
                                truth.actions_defective - original_reference_truth.actions_defective).max())
                            reference_mapping["compensation_max_abs_change"] = float(np.abs(
                                truth.c - original_reference_truth.c).max())
                        dest = result.create_group(f"fields/{label}")
                        for key, value in {"c_source": truth.c, "actions_source": truth.actions_defective,
                            "actions_normal_source": truth.actions_clean, "saturated": truth.clipped,
                            "mapped_states": mapped}.items():
                            dest.create_dataset(key, data=value, compression="gzip")
                        if z is not None:
                            dest.create_dataset("query_defect_state", data=z, compression="gzip")
                        fields[label] = truth.c
                        record.update(status="defined_for_recorded_teacher_under_common_reference", history=semantics, state_mapping=mapping,
                            reference_mapping=reference_mapping,
                            source_delta=meta["delta"], adapter=adapter,
                            recorded_teacher_reproduction_max_abs=reproduction,
                            saturated_fraction=truth.saturated_share,
                            physical_mechanism_validated=source.get("physical_mechanism_validated", None))
                        dest.attrs["definition_json"] = json.dumps(record, sort_keys=True)
                    except UndefinedProbe as exc:
                        record.update(status="undefined", reason=str(exc))
                    finally:
                        if pair is not None:
                            close_pair(pair)
                    records.append(record)
                # Different paired teachers can have different nominal ILC
                # terms. Save the actual discrepancy instead of calling them
                # one common normal reference by assumption.
                references = {}
                if "fields" in result:
                    keys = sorted(result["fields"])
                    for i, first in enumerate(keys):
                        for second in keys[i + 1:]:
                            gap = result[f"fields/{first}/actions_normal_source"][:] - result[f"fields/{second}/actions_normal_source"][:]
                            references[f"{first}__{second}"] = {"max_abs": float(np.abs(gap).max()),
                                "rms": float(np.sqrt(np.mean(gap**2))),
                                "equal_at_1e-6": bool(np.allclose(gap, 0, atol=1e-6, rtol=0))}
                summary = {"schema": SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
                    "anchor": anchor, "anchor_raw_sha256": anchor_sha,
                    "observation_state_checks": observation_checks,
                    "frames": len(states), "fields": records, "field_geometry": field_geometry(fields) if fields else {},
                    "normal_reference_pairwise_gap": references,
                    "all_requested_fields_defined": all(r["status"].startswith("defined") for r in records),
                    "interpretation": "same saved observation/history; each field is a recorded OSC+ILC teacher contrast, not an optimal inverse dynamics solution"}
                result.attrs["summary_json"] = json.dumps(summary, sort_keys=True, allow_nan=False)
            temporary.replace(data_path)
            summary.update(path=str(data_path), sha256=digest(data_path))
            publish(report_path, summary)
            return summary
    finally:
        close_pair(anchor_pair)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--anchor-source", choices=("clean", "delay", "backlash", "droop", "all"), default="clean")
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed-env", type=int, action="append")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args(argv)
    if args.history < 1:
        parser.error("history must be positive")
    os.environ.setdefault("MUJOCO_GL", args.render_backend)
    _, rows = load_manifest(args.manifest.resolve())
    rows = [r for r in rows if r["split"] == args.split and
            (r["source"] == "clean" or r["level"] == args.level) and
            (args.seed_env is None or r["seed_env"] in args.seed_env)]
    anchors = [r for r in rows if args.anchor_source == "all" or r["source"] == args.anchor_source]
    if not anchors:
        raise ValueError("no matching probe anchors")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": SCHEMA, "manifest_sha256": digest(args.manifest),
        "runner_sha256": digest(__file__), "split": args.split, "level": args.level,
        "anchor_source": args.anchor_source, "seed_env": args.seed_env, "history": args.history,
        "undefined_history": "refuse; no zero mechanical state or borrowed other-body history"}
    protocol["protocol_sha256"] = json_hash(protocol)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("output already belongs to another probe protocol")
    publish(protocol_path, protocol)
    reports = []
    for anchor in sorted(anchors, key=lambda r: (r["seed_env"], r["source"], r["level"])):
        sources = [r for r in rows if r["seed_env"] == anchor["seed_env"] and r["source"] != "clean"]
        if {r["source"] for r in sources} != {"delay", "backlash", "droop"} or len(sources) != 3:
            raise ValueError(f"scene {anchor['seed_env']} does not contain exactly three source teachers")
        reports.append(build_one(anchor, sources, args.output, protocol, args.history))
        publish(args.output / "progress.json", {"completed_anchors": len(reports), "planned_anchors": len(anchors)})
    publish(args.output / "summary.json", {"schema": SCHEMA, "status": "complete",
        "anchors": len(reports), "fully_defined_anchors": sum(r["all_requested_fields_defined"] for r in reports),
        "reports": [str(args.output / (Path(r["path"]).stem + ".json")) for r in reports],
        "note": "completion means queries and explicit refusals saved, not that all three physical fields were identifiable"})
    print(json.dumps({"anchors": len(reports), "fully_defined_anchors": sum(r["all_requested_fields_defined"] for r in reports)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
