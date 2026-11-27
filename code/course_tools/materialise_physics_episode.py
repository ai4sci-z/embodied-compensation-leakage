"""Render one physics episode from its complete saved MuJoCo model and states.

This course-only path intentionally does not call the shared finaliser: that
contract describes the earlier equality-based D4hi state, while newer course
episodes use unilateral limit-reaction state semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dl_course.physics_episode_materialisation/2"
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 84
PROPRIO_ACCELERATION_CAUSAL = "causal_velocity_difference_20hz_v1"
PROPRIO_DIM = 70
STATE_SEMANTICS = {
    "d4hi_shadow_equality_efc_pos_pre_step_v1",
    "dl_course.backlash_state.limit-reaction-v1",
}
JOINT_ATOL = 1e-6
EEF_ATOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, str):
        return value
    return None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_xml(demo, root=None) -> str:
    xml = _text(demo.attrs.get("model_file"))
    if xml is None and root is not None:
        xml = _text(root.attrs.get("model_file"))
    if xml is None or "<mujoco" not in xml:
        raise ValueError("raw demo needs the complete MuJoCo XML in attrs['model_file']")
    return xml


def _compile_model(xml: str, state_width: int, demo, location: str) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    if model.nq + model.nv != state_width:
        raise ValueError(
            f"{location}: saved XML has nq={model.nq}, nv={model.nv} "
            f"(nq+nv={model.nq + model.nv}), but states have width {state_width}; "
            "refusing to render potentially shifted joint coordinates"
        )
    joints = [
        {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
            "type": int(model.jnt_type[joint]),
            "qpos_address": int(model.jnt_qposadr[joint]),
            "dof_address": int(model.jnt_dofadr[joint]),
        }
        for joint in range(model.njnt)
    ]
    export_json = _text(demo.attrs.get("physics_export_json"))
    if export_json:
        exported = json.loads(export_json)
        expected = exported.get("state_map")
        if exported.get("model_sha256") != hashlib.sha256(xml.encode()).hexdigest():
            raise ValueError(f"{location}: model_file checksum differs from physics export")
        if expected is not None and (
            expected.get("joints") != joints
            or expected.get("nq") != model.nq
            or expected.get("nv") != model.nv
        ):
            raise ValueError(f"{location}: compiled XML joint map differs from physics export")
    return {"nq": int(model.nq), "nv": int(model.nv), "njnt": int(model.njnt),
            "joints": joints}


def _validate_raw(raw: Path) -> dict[str, Any]:
    import h5py

    summaries: dict[str, Any] = {}
    with h5py.File(raw, "r") as handle:
        if "data" not in handle or not handle["data"]:
            raise ValueError(f"{raw}: missing data demonstrations")
        root_semantics = _text(handle.attrs.get("defect_state_semantics"))
        root_version = handle.attrs.get("defect_state_schema_version")
        for name in handle["data"]:
            demo = handle["data"][name]
            location = f"{raw}:data/{name}"
            semantics = _text(demo.attrs.get("defect_state_semantics", root_semantics))
            version = demo.attrs.get("defect_state_schema_version", root_version)
            if root_semantics is not None and semantics != root_semantics:
                raise ValueError(f"{location}: root/demo defect-state semantics conflict")
            if root_version is not None and version != root_version:
                raise ValueError(f"{location}: root/demo defect-state schema versions conflict")
            if semantics not in STATE_SEMANTICS:
                raise ValueError(
                    f"{location}: unsupported or missing defect-state semantics "
                    f"{semantics!r}; accepted semantics are {sorted(STATE_SEMANTICS)}"
                )
            expected_version = 2 if semantics == "d4hi_shadow_equality_efc_pos_pre_step_v1" else 3
            if version is None or int(version) != expected_version:
                raise ValueError(
                    f"{location}: defect-state schema version {version!r} does not match "
                    f"semantics {semantics!r} (expected {expected_version})"
                )
            if "states" not in demo or demo["states"].ndim != 2:
                raise ValueError(f"{location}: states must be a rank-2 [qpos,qvel] array")
            states = demo["states"]
            if states.shape[0] == 0 or not np.isfinite(states[()]).all():
                raise ValueError(f"{location}: states are empty or non-finite")
            for key in ("actions", "rewards", "dones"):
                if key not in demo or demo[key].shape[0] != states.shape[0]:
                    raise ValueError(f"{location}: {key} must be aligned with states")
            features = demo.get("mechanism_features")
            if features is None:
                raise ValueError(f"{location}: mechanism_features are required for alignment")
            for key in ("joint_position_rad", "joint_velocity_rad_s"):
                if key not in features:
                    raise ValueError(f"{location}: mechanism_features/{key} is required")
                if features[key].shape[0] != states.shape[0]:
                    raise ValueError(f"{location}: mechanism_features/{key} is not frame-aligned")
            for key in ("tau_g", "wrist_ft", "eef_pos_cmd"):
                if key not in demo:
                    raise ValueError(f"{location}: {key} is required to build and verify obs")
                if demo[key].shape[0] != states.shape[0]:
                    raise ValueError(f"{location}: {key} is not frame-aligned with states")
            xml = _read_xml(demo, handle)
            state_map = _compile_model(xml, int(states.shape[1]), demo, location)
            summaries[name] = {
                "frames": int(states.shape[0]),
                "state_width": int(states.shape[1]),
                "state_map": state_map,
                "defect_state_semantics": semantics,
                "defect_state_schema_version": expected_version,
            }
    return summaries


def _add_time_for_converter(raw: Path, temp_input: Path) -> None:
    import h5py

    from portal_trace.registry.tasks import CONTROL_FREQ

    shutil.copyfile(raw, temp_input)
    with h5py.File(temp_input, "r+") as handle:
        for name in handle["data"]:
            demo = handle["data"][name]
            demo.attrs["model_file"] = _read_xml(demo, handle)
            states = np.asarray(demo["states"])
            time = (np.arange(len(states), dtype=states.dtype) / CONTROL_FREQ)[:, None]
            del demo["states"]
            demo.create_dataset("states", data=np.concatenate((time, states), axis=1))
            # The converter must generate fresh pixels from the validated XML.
            if "obs" in demo:
                del demo["obs"]


def _copy_raw_research_fields(raw_demo, out_demo) -> None:
    """Keep every raw research dataset/group without importing raw observations."""
    for name in list(out_demo.keys()):
        if name != "obs":
            del out_demo[name]
    for name in raw_demo:
        if name == "obs":
            continue
        raw_demo.copy(name, out_demo)
    for key, value in raw_demo.attrs.items():
        out_demo.attrs[key] = value


def _finalise_and_check(raw: Path, output: Path, raw_sha: str) -> dict[str, Any]:
    import h5py
    from portal_trace.policies.obs import (
        assemble_proprio,
        control_step_joint_acceleration,
    )

    alignments: dict[str, Any] = {}
    with h5py.File(raw, "r") as source, h5py.File(output, "r+") as target:
        if set(source["data"]) != set(target["data"]):
            raise ValueError("robomimic output demo set differs from the raw episode")
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for key in source:
            if key != "data":
                if key in target:
                    del target[key]
                source.copy(key, target)
        for key, value in source["data"].attrs.items():
            # Converter env_args records the camera setup used for this output.
            if key not in ("env_args", "total"):
                target["data"].attrs[key] = value
        target.attrs["physics_episode_materialisation_schema"] = SCHEMA
        target.attrs["physics_episode_source_sha256"] = raw_sha
        target.attrs["proprio_dim"] = PROPRIO_DIM
        target.attrs["proprio_acceleration_semantics"] = PROPRIO_ACCELERATION_CAUSAL
        for name in sorted(source["data"]):
            raw_demo = source["data"][name]
            out_demo = target["data"][name]
            _copy_raw_research_fields(raw_demo, out_demo)
            obs = out_demo["obs"]
            required_obs = (
                "robot0_joint_pos", "robot0_joint_pos_cos", "robot0_joint_pos_sin",
                "robot0_joint_vel", "robot0_joint_acc",
                "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_site",
                "robot0_gripper_qpos", "robot0_gripper_qvel",
            )
            missing = [key for key in required_obs if key not in obs]
            if missing:
                raise ValueError(f"{output}: converter omitted observations {missing}")
            for camera in CAMERAS:
                key = camera + "_image"
                expected_shape = (len(raw_demo["states"]), IMAGE_SIZE, IMAGE_SIZE, 3)
                if key not in obs or obs[key].shape != expected_shape or obs[key].dtype != np.uint8:
                    raise ValueError(f"{output}: {key} must be uint8 with shape {expected_shape}")

            q_raw = np.asarray(raw_demo["mechanism_features/joint_position_rad"], dtype=np.float64)
            v_raw = np.asarray(raw_demo["mechanism_features/joint_velocity_rad_s"], dtype=np.float64)
            q_obs = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
            v_obs = np.asarray(obs["robot0_joint_vel"], dtype=np.float64)
            if q_raw.shape != q_obs.shape or v_raw.shape != v_obs.shape:
                raise ValueError(
                    f"{raw}: mechanism joint shapes {q_raw.shape}/{v_raw.shape} do not match "
                    f"rendered observations {q_obs.shape}/{v_obs.shape}"
                )
            q_error = np.abs(q_raw - q_obs)
            v_error = np.abs(v_raw - v_obs)
            if not np.isfinite(q_error).all() or not np.isfinite(v_error).all():
                raise ValueError(f"{raw}: non-finite mechanism or rendered joint values")
            if q_error.max(initial=0.0) > JOINT_ATOL or v_error.max(initial=0.0) > JOINT_ATOL:
                raise ValueError(
                    f"{raw}: rendered joint state is misaligned; max position error "
                    f"{q_error.max():.6g}, velocity error {v_error.max():.6g}"
                )

            eef_cmd = np.asarray(raw_demo["eef_pos_cmd"], dtype=np.float64)
            eef_obs = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            if eef_cmd.shape != eef_obs.shape or eef_cmd.shape[-1] != 3:
                raise ValueError(f"{raw}: eef_pos_cmd shape does not match obs/robot0_eef_pos")
            eef_error = np.abs(eef_cmd - eef_obs)
            if not np.isfinite(eef_error).all() or eef_error.max(initial=0.0) > EEF_ATOL:
                raise ValueError(
                    f"{raw}: eef_pos_cmd is not aligned with rendered eef_pos; "
                    f"max error {eef_error.max():.6g} m"
                )

            native = {key: np.asarray(obs[key]) for key in required_obs}
            velocity = native["robot0_joint_vel"]
            acceleration = control_step_joint_acceleration(velocity)
            if len(acceleration) > 1:
                acceleration[1:] = control_step_joint_acceleration(
                    velocity[1:], velocity[:-1]
                )
            del obs["robot0_joint_acc"]
            obs.create_dataset("robot0_joint_acc", data=acceleration, compression="gzip")
            native["robot0_joint_acc"] = acceleration
            proprio = assemble_proprio(
                native,
                tau_g=np.asarray(raw_demo["tau_g"]),
                wrist_ft=np.asarray(raw_demo["wrist_ft"]),
            )
            if proprio.shape != (len(q_obs), PROPRIO_DIM):
                raise ValueError(f"{output}: causal proprioception shape is {proprio.shape}")
            if not np.isfinite(proprio).all():
                raise ValueError(f"{output}: causal proprioception contains non-finite values")
            if "robot0_proprio-state" in obs:
                del obs["robot0_proprio-state"]
            obs.create_dataset("robot0_proprio-state", data=proprio, compression="gzip")

            for key, value in raw_demo.attrs.items():
                out_demo.attrs[key] = value
            out_demo.attrs["proprio_acceleration_semantics"] = PROPRIO_ACCELERATION_CAUSAL
            alignments[name] = {
                "frames": int(len(q_raw)),
                "joint_position_max_abs_error_rad": float(q_error.max(initial=0.0)),
                "joint_velocity_max_abs_error_rad_s": float(v_error.max(initial=0.0)),
                "eef_position_max_abs_error_m": float(eef_error.max(initial=0.0)),
                "joint_tolerance": JOINT_ATOL,
                "eef_tolerance_m": EEF_ATOL,
                "checks": {
                    "mechanism_features_joint_position_equals_obs": True,
                    "mechanism_features_joint_velocity_equals_obs": True,
                    "eef_pos_cmd_equals_obs_eef_pos": True,
                },
                "research_groups": sorted(
                    key for key in raw_demo if hasattr(raw_demo[key], "keys")
                ),
                "research_datasets": sorted(
                    key for key in raw_demo if hasattr(raw_demo[key], "shape")
                ),
            }
    return alignments


def _existing_result(output: Path, report_path: Path, raw: Path) -> dict[str, Any] | None:
    if not output.exists() and not report_path.exists():
        return None
    if not output.exists() or not report_path.exists():
        raise FileExistsError(f"incomplete materialisation already exists for {output}")
    record = json.loads(report_path.read_text(encoding="utf-8"))
    if record.get("source_sha256") != sha256(raw) or record.get("output_sha256") != sha256(output):
        raise ValueError(f"existing materialisation checksums do not match {raw} and {output}")
    if record.get("schema") != SCHEMA or record.get("status") != "passed":
        raise ValueError(f"existing alignment report has an unsupported schema: {report_path}")
    return record


def materialise(raw_path: Path, output_path: Path, *, level: int = 1) -> dict[str, Any]:
    """Render an episode from saved states, preserving raw research fields.

    The output is immutable. A JSON report is written beside it as
    ``<output>.alignment.json`` and contains source/output/code checksums, the
    compiled model dimensions, and per-frame-aligned error maxima.
    """
    raw, output = Path(raw_path).resolve(), Path(output_path).resolve()
    report_path = output.with_suffix(output.suffix + ".alignment.json")
    if raw == output:
        raise ValueError("raw and materialised output paths must differ")
    existing = _existing_result(output, report_path, raw)
    if existing is not None:
        return existing
    if not raw.is_file():
        raise FileNotFoundError(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_sha = sha256(raw)
    model_summaries = _validate_raw(raw)

    input_fd, input_name = tempfile.mkstemp(prefix=".physics-with-time.", suffix=raw.suffix,
                                             dir=output.parent)
    os.close(input_fd)
    render_fd, render_name = tempfile.mkstemp(prefix=f".{output.stem}.rendered.",
                                               suffix=output.suffix, dir=output.parent)
    os.close(render_fd)
    Path(render_name).unlink()
    converter_input, rendered = Path(input_name), Path(render_name)
    replay_audit = rendered.with_suffix(".live-model.json")
    try:
        _add_time_for_converter(raw, converter_input)
        from portal_trace.runners.materialise import MaterialisationSpec

        spec = MaterialisationSpec(raw=converter_input, output=rendered, level=int(level))
        command = spec.command(output_name=rendered.name)
        command[0] = sys.executable
        command[command.index("portal_trace.scripts.materialise_robomimic")] = "course_tools.physics_replay_converter"
        environment = dict(os.environ, COURSE_REPLAY_AUDIT_PATH=str(replay_audit))
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"robomimic state replay failed for {raw}:\n{result.stderr.strip()}"
            )
        if not rendered.is_file():
            raise RuntimeError("robomimic succeeded but did not create its output")
        actual_models = json.loads(replay_audit.read_text())
        alignments = _finalise_and_check(raw, rendered, raw_sha)
        if sha256(raw) != raw_sha:
            raise RuntimeError(f"raw source changed during rendering: {raw}")
        output_sha = sha256(rendered)
        record = {
            "schema": SCHEMA,
            "status": "passed",
            "source_path": str(raw),
            "source_sha256": raw_sha,
            "output_path": str(output),
            "output_sha256": output_sha,
            # The short names are kept for the course collectors' episode
            # manifest convention; the explicit names above remain canonical
            # in the alignment report.
            "raw_path": str(raw),
            "raw_sha256": raw_sha,
            "path": str(output),
            "sha256": output_sha,
            "materialiser_sha256": sha256(Path(__file__)),
            "robomimic_command": command,
            "level": int(level),
            "camera_names": list(CAMERAS),
            "image_size": IMAGE_SIZE,
            "time_alignment": "pre-action state at control index t; converter time=t/20 s",
            "model_dimensions": model_summaries,
            "actual_rendering_models": actual_models,
            "converter_sha256": sha256(Path(__file__).with_name("physics_replay_converter.py")),
            "alignment": alignments,
            "proprio_dim": PROPRIO_DIM,
            "proprio_acceleration_semantics": PROPRIO_ACCELERATION_CAUSAL,
            "rgb_policy": "always re-rendered from raw states and validated model_file; raw obs ignored",
        }
        try:
            os.link(rendered, output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite materialised episode {output}") from None
        _atomic_json(report_path, record)
        return record
    finally:
        converter_input.unlink(missing_ok=True)
        rendered.unlink(missing_ok=True)
        replay_audit.unlink(missing_ok=True)
