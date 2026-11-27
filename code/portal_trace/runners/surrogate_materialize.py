"""Build plan-bound matched surrogates for one held-out evaluation cell."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.defects.fields import FieldContext
from portal_trace.defects.surrogate import (
    SurrogateContext,
    build_surrogate_set,
    construction_hash,
)
from portal_trace.runners.dataset_contract import require_surrogate_context_contract
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    heldout_cell_episode_ids,
    load_formal_data_plan,
    state_manifest_sha256,
)
from portal_trace.runners.surrogate_input import (
    SurrogateInputEpisode,
    SurrogateInputHeader,
    SurrogateInputProvenance,
    write_surrogate_input,
)
from portal_trace.teacher.project import PhiInputs, _kinematics, build_phi

__all__ = ["materialize_surrogate_input"]


_ROBOSUITE_ASSET = re.compile(r'file="[^"]*/robosuite/(?P<relative>[^"]+)"')


@dataclass(frozen=True)
class _EpisodeMaterial:
    episode_id: str
    inputs: PhiInputs
    context: FieldContext
    c_true: np.ndarray
    a_nom: np.ndarray
    kinematics: dict[str, np.ndarray]

    @property
    def n_frames(self) -> int:
        return len(self.c_true)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _model_from_stored_xml(xml: str) -> tuple[Any, Any]:
    """Compile stored robosuite XML after relocating package-owned assets."""
    import mujoco
    import robosuite

    package_root = Path(robosuite.__file__).resolve().parent

    def relocate(match: re.Match[str]) -> str:
        return f'file="{package_root / match.group("relative")}"'

    model = mujoco.MjModel.from_xml_string(_ROBOSUITE_ASSET.sub(relocate, xml))
    return model, mujoco.MjData(model)


def _episode_row(plan: FormalDataPlan, episode_id: str) -> dict[str, Any]:
    rows = [row for row in plan.payload["pool"] if row["id"] == episode_id]
    if len(rows) != 1:
        raise ValueError(f"formal plan does not bind exactly one row for {episode_id}")
    return rows[0]


def _load_episode(plan: FormalDataPlan, episode_id: str) -> _EpisodeMaterial:
    import h5py

    row = _episode_row(plan, episode_id)
    path = Path(row["path"])
    try:
        path_text, demo_name = episode_id.rsplit("::data/", 1)
    except ValueError as exc:
        raise ValueError(f"malformed formal episode ID {episode_id!r}") from exc
    if Path(path_text).resolve() != path.resolve() or demo_name != row["demo"]:
        raise ValueError(f"{episode_id}: plan path/demo binding changed")
    with h5py.File(path, "r") as root:
        demo = root["data"][demo_name]
        context_group = demo["surrogate_context"]
        horizon = len(demo["actions"])
        analysis_horizon = horizon - 1
        context = FieldContext(
            qpos_index=tuple(int(value) for value in context_group["qpos_index"][()]),
            qvel_index=tuple(int(value) for value in context_group["qvel_index"][()]),
            site_id=int(context_group.attrs["site_id"]),
            kp=np.asarray(context_group["kp"], dtype=np.float64),
            output_max=np.asarray(context_group["output_max"], dtype=np.float64),
            action_dim=int(context_group.attrs["action_dim"]),
        )
        injected_joints = tuple(
            int(value) for value in context_group["injected_joints"][()]
        )
        states = np.asarray(demo["states"][:analysis_horizon], dtype=np.float64)
        c_true = np.asarray(demo["c_true"][:analysis_horizon], dtype=np.float64)
        a_nom = np.asarray(
            demo["actions_clean"][:analysis_horizon], dtype=np.float64
        )
        phi_inputs_data = {
            "states": states,
            "u_inf": np.asarray(demo["u_inf"][:analysis_horizon], dtype=np.float64),
            "regime": np.asarray(demo["regime"][:analysis_horizon], dtype=np.uint8),
            "task": np.full(analysis_horizon, int(row["task_index"]), dtype=np.int64),
            "phase": np.asarray(demo["phase"][:analysis_horizon], dtype=np.uint8),
            "a_dot_nominal": np.asarray(
                context_group["a_dot_nominal"], dtype=np.float64
            ),
            "joint_torque": np.asarray(
                demo["joint_torque"][:analysis_horizon], dtype=np.float64
            ),
            "contact": np.asarray(
                demo["contact"][:analysis_horizon], dtype=bool
            ),
            "injected_joints": injected_joints,
        }
        model_xml = _decode(demo.attrs["model_file"])
    model, data = _model_from_stored_xml(model_xml)
    if states.shape != (analysis_horizon, int(model.nq + model.nv)):
        raise ValueError(
            f"{episode_id}: states have shape {states.shape}, expected "
            f"({analysis_horizon}, {model.nq + model.nv}) for the stored model"
        )
    if (
        context.site_id >= model.nsite
        or max(context.qpos_index) >= model.nq
        or max(context.qvel_index) >= model.nv
    ):
        raise ValueError(f"{episode_id}: stored kinematic indices exceed the model")
    inputs = PhiInputs(model=model, data=data, ctx=context, **phi_inputs_data)
    kinematics = _kinematics(inputs)
    if c_true.shape != (analysis_horizon, context.action_dim):
        raise ValueError(f"{episode_id}: c_true shape disagrees with surrogate context")
    if a_nom.shape != c_true.shape:
        raise ValueError(f"{episode_id}: clean action shape disagrees with c_true")
    if not all(
        np.isfinite(values).all()
        for values in (states, c_true, a_nom, *kinematics.values())
    ):
        raise ValueError(f"{episode_id}: surrogate source contains non-finite values")
    return _EpisodeMaterial(
        episode_id=episode_id,
        inputs=inputs,
        context=context,
        c_true=c_true,
        a_nom=a_nom,
        kinematics=kinematics,
    )


def _same_context(left: FieldContext, right: FieldContext) -> bool:
    return (
        left.qpos_index == right.qpos_index
        and left.qvel_index == right.qvel_index
        and left.site_id == right.site_id
        and left.action_dim == right.action_dim
        and np.array_equal(left.kp, right.kp)
        and np.array_equal(left.output_max, right.output_max)
    )


def _concatenate(material: tuple[_EpisodeMaterial, ...]) -> SurrogateContext:
    first = material[0]
    if any(not _same_context(first.context, item.context) for item in material[1:]):
        raise ValueError("surrogate context changed between episodes in one cell")
    injected = first.inputs.injected_joints
    if any(item.inputs.injected_joints != injected for item in material[1:]):
        raise ValueError("injected joint set changed between episodes in one cell")

    def join_input(name: str) -> np.ndarray:
        return np.concatenate([getattr(item.inputs, name) for item in material], axis=0)

    cell_inputs = PhiInputs(
        states=join_input("states"),
        u_inf=join_input("u_inf"),
        regime=join_input("regime"),
        task=join_input("task"),
        phase=join_input("phase"),
        a_dot_nominal=join_input("a_dot_nominal"),
        joint_torque=join_input("joint_torque"),
        contact=join_input("contact"),
        injected_joints=injected,
        model=first.inputs.model,
        data=first.inputs.data,
        ctx=first.context,
    )
    kinematics = {
        name: np.concatenate([item.kinematics[name] for item in material], axis=0)
        for name in first.kinematics
    }
    phi = build_phi(cell_inputs, kinematics=kinematics).raw
    return SurrogateContext(
        c_true=np.concatenate([item.c_true for item in material], axis=0),
        episode_index=np.concatenate(
            [np.full(item.n_frames, index, dtype=np.int64) for index, item in enumerate(material)]
        ),
        regime=cell_inputs.regime,
        jacobian=kinematics["jac"],
        qvel=kinematics["qvel"],
        ee_pos=kinematics["eef_pos"],
        ee_vel=kinematics["twist"][:, :3],
        output_max=first.context.output_max,
        a_nom=np.concatenate([item.a_nom for item in material], axis=0),
        phi=phi,
    )


def _finite_report(report: dict[str, Any]) -> dict[str, Any]:
    """Encode undefined match metrics explicitly instead of emitting invalid JSON."""
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in report.items()
    }


def materialize_surrogate_input(
    *,
    output: Path,
    evaluation_plan_path: Path,
    task: str,
    family: str,
    defect: str,
    level: int,
    seed: int,
    allow_engineering: bool = False,
) -> SurrogateInputHeader:
    """Generate the eight closed forms and publish the admitted fields."""
    from portal_trace.runners.formal_training import _git_training_producer_state

    if defect == "clean":
        raise ValueError("matched surrogates require a defective evaluation cell")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("surrogate seed must be an integer")
    plan = load_formal_data_plan(Path(evaluation_plan_path))
    episode_ids = heldout_cell_episode_ids(
        plan, task=task, family=family, defect=defect, level=level
    )
    paths = {Path(_episode_row(plan, episode_id)["path"]) for episode_id in episode_ids}
    for path in paths:
        require_surrogate_context_contract(path)

    producer_start = _git_training_producer_state()
    if not allow_engineering and producer_start["tracked_worktree_dirty"]:
        raise RuntimeError("formal surrogate materialisation requires a clean tracked HEAD")
    material = tuple(_load_episode(plan, episode_id) for episode_id in episode_ids)
    context = _concatenate(material)
    surrogate_set = build_surrogate_set(
        context,
        cell=f"{task}/{family}/{defect}/L{level}",
        seed=seed,
    )
    producer_end = _git_training_producer_state()
    if producer_start != producer_end:
        raise RuntimeError("tracked source changed during surrogate materialisation")

    admitted = surrogate_set.admitted
    if not admitted:
        raise ValueError("the held-out cell admitted no matched surrogate")
    fields = surrogate_set.null_fields()
    episodes: list[SurrogateInputEpisode] = []
    offset = 0
    for item in material:
        stop = offset + item.n_frames
        episodes.append(
            SurrogateInputEpisode(
                episode_id=item.episode_id,
                fields={name: values[offset:stop] for name, values in fields.items()},
            )
        )
        offset = stop
    if offset != context.n_frames:
        raise AssertionError("surrogate episode split did not consume the cell frames")
    reports = {
        name: _finite_report(
            {
                "beta": result.beta,
                "rounds": result.rounds,
                "joint_rotation_trials": result.joint_rotation_trials,
                **result.report.to_json(),
            }
        )
        for name, result in sorted(surrogate_set.results.items())
    }
    provenance = SurrogateInputProvenance(
        evaluation_plan_sha256=plan.plan_sha256,
        source_state_manifest_sha256=state_manifest_sha256(plan, episode_ids),
        construction_sha256=construction_hash(),
        task=task,
        family=family,
        defect=defect,
        level=level,
        seed=seed,
        admitted_surrogates=admitted,
        struck_surrogates=surrogate_set.struck,
        match_reports=reports,
        release_status="engineering_only" if allow_engineering else "formal",
    )
    return write_surrogate_input(output, provenance=provenance, episodes=episodes)
