"""Image materialisation (proposal 9.1.3, 9.6.2, 9.6.3).

Rendering is decoupled from collection. 9.1.3 measures rendering at roughly
eight times the cost of physics, and the same states are consumed by several
arms, so demonstrations are collected once without pixels and images are
produced from the stored states on demand. Materialising per arm instead would
repeat the most expensive step once per arm.

This stage is a thin, pinned wrapper around robomimic 0.5.0's
`dataset_states_to_obs`. It is a wrapper rather than a reimplementation because
the observation layer has to be byte-identical to what robomimic's training
loop expects; what this module adds is that every flag which changes the
produced tensors is pinned here and recorded, and that the width of the stored
proprioceptive vector is checked against the measured layout instead of being
assumed.

CONTRACT section 6 overrides 9.6.3 on that width: robosuite 1.5.2's native
`robot0_proprio-state` is 50-dimensional on a 7-DoF arm, so the stored vector is
70 = 50 + 7 tau_g + 6 wrist F/T + 7 DoF mask, not the 52 the proposal prints.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from portal_trace.runners.collect import PROPRIO_DIM, PROPRIO_LAYOUT
from portal_trace.runners.common import (
    RunContext,
    dataset_paths,
    reject_legacy_dataset,
    require_level_dataset,
)
from portal_trace.runners.dataset_contract import (
    require_analysis_trace_contract,
    require_current_defect_state_contract,
    require_dataset_identity,
    require_delta_value_contract,
)

__all__ = [
    "MaterialisationSpec",
    "materialise",
    "materialise_b7_clean",
    "verify_proprio_width",
    "finalise_materialised_dataset",
    "CAMERAS",
    "IMAGE_SIZE",
]

# 9.6.2: two 84x84 cameras. agentview supplies the scene, robot0_eye_in_hand
# the wrist view; the pair is what the 9.8.3 storage line is computed on.
CAMERAS: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 84


@dataclass(frozen=True)
class MaterialisationSpec:
    """One raw file's rendering job, with every tensor-changing flag pinned."""

    raw: Path
    output: Path
    level: int
    cameras: tuple[str, ...] = CAMERAS
    image_size: int = IMAGE_SIZE
    compress: bool = True
    exclude_next_obs: bool = True
    done_mode: int = 2

    def command(self, *, output_name: str | None = None) -> list[str]:
        """The robomimic invocation, with no flag left implicit.

        `--done_mode 2` and `--exclude-next-obs` are pinned rather than left at
        robomimic's defaults: both change the stored tensors, and a dataset
        rendered under one setting cannot be pooled with one rendered under the
        other.
        """
        cmd = [
            "python",
            "-m",
            "portal_trace.scripts.materialise_robomimic",
            "--dataset", str(self.raw),
            # robomimic always joins output_name to dirname(dataset). The
            # caller therefore gives it a temporary basename beside `raw` and
            # publishes that completed file into the separate obs tree.
            "--output_name", output_name or self.output.name,
            "--done_mode", str(self.done_mode),
            "--camera_names", *self.cameras,
            "--camera_height", str(self.image_size),
            "--camera_width", str(self.image_size),
            # The raw trajectory is authoritative for both fields. Inferred
            # values would introduce a robosuite-version-dependent data change.
            "--copy_rewards",
            "--copy_dones",
        ]
        if self.compress:
            cmd.append("--compress")
        if self.exclude_next_obs:
            cmd.append("--exclude-next-obs")
        return cmd


def verify_proprio_width(path: Path) -> int:
    """Check the stored proprioceptive width against the measured layout.

    Raises rather than warning. The width appears in every model card and in the
    storage total; a silently different one means the observation the policy saw
    is not the observation the card describes.
    """
    import h5py

    with h5py.File(path, "r") as fh:
        demos = list(fh["data"].keys())
        if not demos:
            raise ValueError(f"{path}: no demonstrations in the materialised file")
        obs = fh["data"][demos[0]]["obs"]
        if "robot0_proprio-state" not in obs:
            raise ValueError(
                f"{path}: no robot0_proprio-state in obs/. The 9.6.3 layout stores "
                "the proprioceptive vector explicitly; a file without it cannot be "
                "used for a preregistered endpoint"
            )
        width = int(obs["robot0_proprio-state"].shape[-1])
    if width != PROPRIO_DIM:
        layout = ", ".join(f"{name} = {n}" for name, n in PROPRIO_LAYOUT)
        raise ValueError(
            f"{path}: robot0_proprio-state is {width} wide, the measured layout gives "
            f"{PROPRIO_DIM} ({layout}).\n"
            "This width is measured, not assumed: robosuite 1.5.2's native vector is "
            "50 on a 7-DoF arm, which is why the proposal's 52 is corrected to 70 in "
            "the amendment log. A different width here means either a different arm "
            "DoF or a different set of appended channels, and both change what the "
            "policy sees."
        )
    return width


def finalise_materialised_dataset(raw: Path, output: Path) -> None:
    """Attach PORTAL-Trace fields that robomimic's converter does not know.

    Robomimic writes its native 50-wide ``robot0_proprio-state`` and omits all
    project-specific datasets. The training file must instead carry the explicit
    70-wide cross-family layout plus the ground-truth columns used by B8 and the
    diagnostics. Both files are opened together so demo identity cannot drift.
    """
    import h5py

    from portal_trace.policies.obs import (
        assemble_proprio,
        control_step_joint_acceleration,
        PROPRIO_ACCELERATION_CAUSAL,
    )

    require_current_defect_state_contract(raw)
    require_analysis_trace_contract(raw)

    extensions = (
        "delta_true",
        "actions_defective",
        "actions_clean",
        "c_true",
        "c_true_clipped",
        "defect_state",
        "actuation_state",
        "regime",
        "u_inf",
        "u_inf_clean",
        "phase",
        "contact",
        "eef_pos_cmd",
        "eef_pos_ach",
        "tracking_error",
        "joint_torque",
    )
    with h5py.File(raw, "r") as source, h5py.File(output, "r+") as target:
        source_demos = set(source["data"])
        target_demos = set(target["data"])
        if source_demos != target_demos:
            raise ValueError(
                f"materialised demos do not match raw demos: raw-only "
                f"{sorted(source_demos - target_demos)}, output-only "
                f"{sorted(target_demos - source_demos)}"
            )
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["proprio_dim"] = PROPRIO_DIM
        target.attrs["proprio_acceleration_semantics"] = PROPRIO_ACCELERATION_CAUSAL

        for demo in sorted(source_demos, key=lambda key: int(key.split("_")[1])):
            raw_demo = source["data"][demo]
            out_demo = target["data"][demo]
            for signal in ("tau_g", "wrist_ft", *extensions):
                if signal not in raw_demo:
                    raise ValueError(
                        f"{raw}: data/{demo}/{signal} is missing; materialisation "
                        "cannot reconstruct the declared PORTAL-Trace data contract"
                    )
            obs = out_demo["obs"]
            native = {
                key: obs[key][()]
                for key in (
                    "robot0_joint_pos",
                    "robot0_joint_pos_cos",
                    "robot0_joint_pos_sin",
                    "robot0_joint_vel",
                    "robot0_joint_acc",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_eef_quat_site",
                    "robot0_gripper_qpos",
                    "robot0_gripper_qvel",
                )
            }
            # State replay restores qpos/qvel, but not the live actuator force
            # that produced qacc. Use the same causal velocity difference as
            # deployment, before any histories are sliced from the episode.
            velocity = native["robot0_joint_vel"]
            acceleration = control_step_joint_acceleration(velocity)
            acceleration[1:] = control_step_joint_acceleration(
                velocity[1:], velocity[:-1]
            )
            native["robot0_joint_acc"] = acceleration
            del obs["robot0_joint_acc"]
            obs.create_dataset(
                "robot0_joint_acc", data=acceleration, compression="gzip"
            )
            combined = assemble_proprio(
                native, tau_g=raw_demo["tau_g"][()], wrist_ft=raw_demo["wrist_ft"][()]
            )
            # The converter input receives a temporary leading time column for
            # robosuite replay.  The published state remains the raw (qpos, qvel)
            # vector so model-bound consumers see exactly nq + nv columns.
            del out_demo["states"]
            out_demo.create_dataset(
                "states", data=raw_demo["states"][()], compression="gzip"
            )
            if "robot0_proprio-state" in obs:
                del obs["robot0_proprio-state"]
            obs.create_dataset(
                "robot0_proprio-state", data=combined, compression="gzip"
            )

            for name in extensions:
                if name in out_demo:
                    del out_demo[name]
                out_demo.create_dataset(
                    name, data=raw_demo[name][()], compression="gzip"
                )
            if "surrogate_context" in raw_demo:
                if "surrogate_context" in out_demo:
                    del out_demo["surrogate_context"]
                raw_demo.copy("surrogate_context", out_demo)
            for key, value in raw_demo.attrs.items():
                out_demo.attrs[key] = value


def materialise(ctx: RunContext, *, check_only: bool = False) -> MaterialisationSpec:
    """Render one cell's observations from its stored states."""
    req = ctx.request
    if req.task is None or req.family is None or req.defect is None or req.level is None:
        raise ValueError(
            "materialisation needs --task, --family, --defect and --level: the dose "
            "level is part of the raw and observation dataset identity"
        )
    level = int(req.level)
    paths = dataset_paths(
        ctx.dataset_root,
        task=req.task,
        family=req.family,
        defect=req.defect,
        level=level,
    )
    spec = MaterialisationSpec(raw=paths["raw"], output=paths["obs"], level=level)
    require_level_dataset(
        spec.raw,
        produced_by=(
            "collection has to run for this exact dose level before materialisation "
            "(9.1.3)"
        ),
    )
    require_dataset_identity(
        spec.raw,
        task=req.task,
        family=req.family,
        defect=req.defect,
        level=level,
    )
    require_current_defect_state_contract(spec.raw, requested_defect=req.defect)
    require_delta_value_contract(spec.raw)
    reject_legacy_dataset(spec.output)

    clean_paths = dataset_paths(
        ctx.dataset_root,
        task=req.task,
        family=req.family,
        defect="clean",
        level=level,
        clean_of=req.defect,
    )
    clean_spec = MaterialisationSpec(
        raw=clean_paths["raw"], output=clean_paths["obs"], level=level
    )
    require_level_dataset(
        clean_spec.raw,
        produced_by=(
            "twin collection must produce the defect-specific clean companion before "
            "materialisation"
        ),
    )
    require_dataset_identity(
        clean_spec.raw,
        task=req.task,
        family=req.family,
        defect="clean",
        level=level,
        clean_of=req.defect,
    )
    require_current_defect_state_contract(
        clean_spec.raw, requested_defect=req.defect
    )
    require_delta_value_contract(clean_spec.raw)
    # Analysis traces are collected at the exact control-step boundary and
    # cannot be reconstructed by the image renderer.  Check both raw members
    # after their defect-state semantics, but before invoking the external
    # converter, so an old B7 part/assembly fails closed without render work.
    require_analysis_trace_contract(spec.raw)
    require_analysis_trace_contract(clean_spec.raw)
    reject_legacy_dataset(clean_spec.output)

    if check_only:
        require_level_dataset(
            spec.output,
            produced_by="materialisation has not produced this dose level",
        )
        require_dataset_identity(
            spec.output,
            task=req.task,
            family=req.family,
            defect=req.defect,
            level=level,
        )
        require_current_defect_state_contract(
            spec.output, requested_defect=req.defect
        )
        require_analysis_trace_contract(spec.output)
        verify_proprio_width(spec.output)
        require_level_dataset(
            clean_spec.output,
            produced_by="materialisation has not produced this clean companion level",
        )
        require_dataset_identity(
            clean_spec.output,
            task=req.task,
            family=req.family,
            defect="clean",
            level=level,
            clean_of=req.defect,
        )
        require_current_defect_state_contract(
            clean_spec.output, requested_defect=req.defect
        )
        require_analysis_trace_contract(clean_spec.output)
        verify_proprio_width(clean_spec.output)
        return spec

    # Validate both members and both immutable destinations before rendering
    # either one. Otherwise a bad or missing clean companion could strand a
    # published defective output that the retry is then forbidden to replace.
    for output in (spec.output, clean_spec.output):
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite materialised dataset {output}"
            )

    _materialise_one(spec)
    _materialise_one(clean_spec)
    return spec


def materialise_b7_clean(
    ctx: RunContext, *, check_only: bool = False
) -> MaterialisationSpec:
    """Render a B7 clean-only raw pool into the policy observation layout.

    ``collect-b7-clean`` intentionally produces only the clean twin.  The
    ordinary :func:`materialise` stage requires a defective/clean pair because
    it serves paired collection, so using it after B7 clean-only collection
    would fail before rendering.  This explicit entry point keeps that pair
    requirement intact while admitting the clean-only B7 producer.
    """
    req = ctx.request
    if req.task is None or req.family is None or req.defect is None or req.level is None:
        raise ValueError(
            "B7 clean materialisation needs --task, --family, --defect and --level"
        )
    from portal_trace.registry.tasks import CORE_TASKS

    if req.task not in CORE_TASKS:
        raise ValueError(
            f"B7 clean materialisation is defined only for core tasks, got {req.task!r}"
        )
    if (req.family, req.defect) not in set(ctx.bundle.cells("B7")):
        raise ValueError(
            "B7 clean materialisation requires a declared B7 target cell, got "
            f"({req.family}, {req.defect})"
        )
    level = int(req.level)
    paths = dataset_paths(
        ctx.dataset_root,
        task=req.task,
        family=req.family,
        defect="clean",
        level=level,
        clean_of=req.defect,
    )
    spec = MaterialisationSpec(raw=paths["raw"], output=paths["obs"], level=level)
    require_level_dataset(
        spec.raw,
        produced_by=(
            "collect-b7-clean must produce the complete clean raw pool before "
            "B7 materialisation"
        ),
    )
    require_dataset_identity(
        spec.raw,
        task=req.task,
        family=req.family,
        defect="clean",
        level=level,
        clean_of=req.defect,
    )
    require_current_defect_state_contract(spec.raw, requested_defect=req.defect)
    require_delta_value_contract(spec.raw)
    require_analysis_trace_contract(spec.raw)
    reject_legacy_dataset(spec.output)

    if check_only:
        require_level_dataset(
            spec.output,
            produced_by="B7 clean materialisation has not produced this level",
        )
        require_dataset_identity(
            spec.output,
            task=req.task,
            family=req.family,
            defect="clean",
            level=level,
            clean_of=req.defect,
        )
        require_current_defect_state_contract(spec.output, requested_defect=req.defect)
        require_analysis_trace_contract(spec.output)
        verify_proprio_width(spec.output)
        return spec

    _materialise_one(spec)
    return spec


def _materialise_one(spec: MaterialisationSpec) -> None:
    """Run and validate one raw -> obs conversion with pinned flags."""
    if spec.output.exists():
        raise FileExistsError(
            f"refusing to overwrite materialised dataset {spec.output}"
        )
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=spec.raw.parent,
        prefix=f".{spec.output.stem}.",
        suffix=spec.output.suffix,
    )
    os.close(fd)
    Path(temporary_name).unlink()
    converter_fd, converter_name = tempfile.mkstemp(
        dir=spec.raw.parent,
        prefix=f".{spec.raw.stem}.with-time.",
        suffix=spec.raw.suffix,
    )
    os.close(converter_fd)
    converter_input = Path(converter_name)
    shutil.copyfile(spec.raw, converter_input)
    try:
        # PORTAL-Trace stores (qpos, qvel), while robosuite's flattened state
        # parser requires [time, qpos, qvel]. Add only the deterministic control
        # clock to this private converter copy; the raw contract remains intact.
        import h5py

        from portal_trace.registry.tasks import CONTROL_FREQ

        with h5py.File(converter_input, "r+") as source:
            for name in source["data"]:
                demo = source["data"][name]
                states = demo["states"][()]
                times = (np.arange(states.shape[0], dtype=states.dtype) / CONTROL_FREQ)[:, None]
                del demo["states"]
                demo.create_dataset("states", data=np.concatenate((times, states), axis=1))
        converter_spec = MaterialisationSpec(
            raw=converter_input,
            output=spec.output,
            level=spec.level,
            cameras=spec.cameras,
            image_size=spec.image_size,
            compress=spec.compress,
            exclude_next_obs=spec.exclude_next_obs,
            done_mode=spec.done_mode,
        )
        # The converter expects to create the output itself.
        temporary = Path(temporary_name)
        result = subprocess.run(
            converter_spec.command(output_name=temporary.name), capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"robomimic's dataset_states_to_obs failed on {spec.raw}:\n"
                f"{result.stderr.strip()}"
            )
        if not temporary.exists():
            raise RuntimeError(
                "robomimic reported success but did not create its declared output "
                f"{temporary}"
            )
        finalise_materialised_dataset(spec.raw, temporary)
        verify_proprio_width(temporary)
        try:
            os.link(temporary, spec.output)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite materialised dataset {spec.output}"
            ) from None
    finally:
        Path(temporary_name).unlink(missing_ok=True)
        converter_input.unlink(missing_ok=True)
