"""Course-only mechanical backlash, version ``unilateral-limit-v1``.

The two co-located coordinates are motor angle q and relative angle s. The
output angle is q+s. The motor's existing armature supplies its rotor inertia;
the downstream links supply output inertia. There is no coupling torque for
-b/2 < s < b/2. Unilateral joint limits transmit torque at the two boundaries.
This is a regularized contact model, not an action deadband or a centering
spring. The mechanical history (s, sdot) lives in qpos/qvel and must be saved.

Nothing here monkey-patches portal_trace. Call ``make_pair`` or
``make_wrapper`` explicitly, and stamp ``PHYSICS_VERSION`` on new datasets.
The disabled legacy equality is retained only because the base wrapper resolves
its ID; neither its residual nor its force defines this version's engagement.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PHYSICS_VERSION = "dl_course.backlash.unilateral-limit-v1"
STATE_CONTRACT = "dl_course.backlash_state.limit-reaction-v1"
LIMIT_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)
TRANSMITTING_TORQUE_TOLERANCE_NM = 1e-9


def limit_solref(timestep: float) -> tuple[float, float]:
    """Use MuJoCo's safe minimum time constant, with critical damping."""
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("physics timestep must be finite and positive")
    return (2.0 * timestep, 1.0)


def rewrite_backlash_xml(xml: str, specs, *, timestep: float) -> str:
    """Configure already inserted shadow joints; preserve structural indexing."""
    root = ET.fromstring(xml)
    joints = {j.get("name"): j for b in root.iter("body") for j in b.findall("joint")}
    equality = root.find("equality")
    equalities = {} if equality is None else {e.get("name"): e for e in equality}
    for spec in specs:
        half = float(spec.half_range)
        if not np.isfinite(half) or half <= 0:
            raise ValueError("backlash half width must be a positive number of radians")
        node = joints[spec.name]
        node.attrib.update(limited="true", range=f"{-half:.17g} {half:.17g}",
                           ref="0", springref="0", stiffness="0", damping="0",
                           frictionloss="0", armature="0", margin="0",
                           solreflimit=" ".join(map(str, limit_solref(timestep))),
                           solimplimit=" ".join(map(str, LIMIT_SOLIMP)))
        equalities[spec.equality_name].set("active", "false")
    return ET.tostring(root, encoding="unicode")


def configure_compiled_model(model, data, shadow_ids, equality_ids) -> None:
    """Reapply after hard reset; force explicit defaults even on inherited MJCF."""
    for jid in shadow_ids:
        dof = int(model.jnt_dofadr[jid])
        model.jnt_limited[jid] = True
        model.jnt_margin[jid] = 0
        model.jnt_stiffness[jid] = 0
        model.dof_damping[dof] = 0
        model.dof_frictionloss[dof] = 0
        model.dof_armature[dof] = 0
        model.jnt_solref[jid] = limit_solref(float(model.opt.timestep))
        model.jnt_solimp[jid] = LIMIT_SOLIMP
    for eid in equality_ids:
        model.eq_active0[eid] = False
        data.eq_active[eid] = False
    mujoco.mj_forward(model, data)


@dataclass(frozen=True)
class Transmission:
    input_name: str
    shadow_name: str
    input_joint_id: int
    shadow_joint_id: int
    input_qpos: int
    shadow_qpos: int
    input_dof: int
    shadow_dof: int
    half_width_rad: float

    @classmethod
    def resolve(cls, model, input_name, shadow_name):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in (input_name, shadow_name)]
        if min(ids) < 0:
            raise ValueError(f"missing transmission joints: {input_name}, {shadow_name}")
        a, b = ids
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
            raise ValueError("mechanical backlash requires two hinge coordinates")
        if (model.jnt_bodyid[a] != model.jnt_bodyid[b]
                or not np.allclose(model.jnt_axis[a], model.jnt_axis[b], atol=1e-12)
                or not np.allclose(model.jnt_pos[a], model.jnt_pos[b], atol=1e-12)):
            raise ValueError("input and shadow must be co-located, co-axial, in one body")
        if model.dof_armature[int(model.jnt_dofadr[a])] <= 0:
            raise ValueError("input joint requires positive rotor armature")
        half = float(model.jnt_range[b, 1])
        if not model.jnt_limited[b] or half <= 0 or model.jnt_range[b, 0] != -half:
            raise ValueError("shadow requires symmetric nonzero limits in radians")
        return cls(input_name, shadow_name, a, b, int(model.jnt_qposadr[a]),
                   int(model.jnt_qposadr[b]), int(model.jnt_dofadr[a]),
                   int(model.jnt_dofadr[b]), half)


def transmission_diagnostics(model, data, transmission: Transmission) -> dict:
    """Read forces from a completed mj_forward at the current state/control.

    Limit force is projected through efc_J, so its sign is the generalized
    torque on the relative coordinate. Since s=q_out-q_in, this is also the
    torque transmitted to the output; the opposite acts on the input rotor.
    Other robot contacts/limits are excluded from this transmission torque.
    """
    t = transmission
    n = int(data.nefc)
    rows = np.flatnonzero((np.asarray(data.efc_type[:n]) ==
                          int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) &
                         (np.asarray(data.efc_id[:n]) == t.shadow_joint_id))
    filtered = np.zeros(n)
    filtered[rows] = np.asarray(data.efc_force[:n])[rows]
    projected = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, projected, filtered)
    torque = float(projected[t.shadow_dof])
    s = float(data.qpos[t.shadow_qpos])
    boundary = int(np.sign(s)) if rows.size else 0
    transmitting = bool(rows.size and abs(torque) > TRANSMITTING_TORQUE_TOLERANCE_NM)
    qi, vi = float(data.qpos[t.input_qpos]), float(data.qvel[t.input_dof])
    vs = float(data.qvel[t.shadow_dof])
    return {
        "time_s": float(data.time), "input_position_rad": qi,
        "output_position_rad": qi + s, "relative_position_rad": s,
        "input_velocity_rad_s": vi, "output_velocity_rad_s": vi + vs,
        "relative_velocity_rad_s": vs, "total_width_rad": 2*t.half_width_rad,
        "gap_to_lower_rad": s + t.half_width_rad,
        "gap_to_upper_rad": t.half_width_rad - s,
        "penetration_rad": max(0.0, abs(s) - t.half_width_rad),
        "limit_active": bool(rows.size), "limit_rows": int(rows.size),
        "boundary_side": boundary, "transmitting": transmitting,
        "transmission_torque_nm": torque,
        "input_actuator_torque_nm": float(data.qfrc_actuator[t.input_dof]),
        "shadow_passive_torque_nm": float(data.qfrc_passive[t.shadow_dof]),
        "limit_residual_rad": [float(data.efc_pos[r]) for r in rows],
        "limit_force_nm": [float(data.efc_force[r]) for r in rows],
    }


def model_state_map(model) -> dict:
    return {"nq": int(model.nq), "nv": int(model.nv), "na": int(model.na),
            "joints": [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                        "type": int(model.jnt_type[j]),
                        "qpos_address": int(model.jnt_qposadr[j]),
                        "dof_address": int(model.jnt_dofadr[j])}
                       for j in range(model.njnt)]}


def compiled_model_xml(model) -> str:
    """Export the actual compiled model, not robosuite's pre-injection XML."""
    with tempfile.TemporaryDirectory(prefix="backlash-model-") as directory:
        path = Path(directory) / "compiled.xml"
        mujoco.mj_saveLastXML(str(path), model)
        # MuJoCo's writer rounds floats to six significant digits. A 0.1-degree
        # limit needs its exact width for deterministic force/state replay.
        root = ET.fromstring(path.read_text())
        for body in root.iter("body"):
            for joint in body.findall("joint"):
                name = joint.get("name")
                if not name:
                    continue
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                dof = int(model.jnt_dofadr[jid])
                for attr, value in {
                    "range": model.jnt_range[jid], "margin": [model.jnt_margin[jid]],
                    "stiffness": [model.jnt_stiffness[jid]],
                    "armature": [model.dof_armature[dof]],
                    "damping": [model.dof_damping[dof]],
                    "frictionloss": [model.dof_frictionloss[dof]],
                    "solreflimit": model.jnt_solref[jid],
                    "solimplimit": model.jnt_solimp[jid],
                }.items():
                    joint.set(attr, " ".join(f"{float(x):.17g}" for x in value))
        return ET.tostring(root, encoding="unicode")


def verify_exported_model(model, xml: str) -> dict:
    restored = mujoco.MjModel.from_xml_string(xml)
    actual, expected = model_state_map(restored), model_state_map(model)
    if actual != expected:
        raise ValueError("exported XML changed qpos/qvel joint mapping")
    for name in ("jnt_range", "jnt_margin", "jnt_stiffness", "jnt_solref", "jnt_solimp",
                 "dof_damping", "dof_frictionloss", "dof_armature", "eq_active0"):
        if not np.allclose(getattr(model, name), getattr(restored, name), rtol=1e-5, atol=1e-10):
            raise ValueError(f"exported XML changed {name}")
    return expected


@lru_cache(maxsize=1)
def backlash_wrapper_class():
    """Import robosuite only for full robot construction, not the small probe."""
    from portal_trace.defects.inject import DefectWrapper

    class LimitBacklashWrapper(DefectWrapper):
        physics_version = PHYSICS_VERSION
        defect_state_contract = STATE_CONTRACT

        def __init__(self, env, delta, *, seeds, family):
            if delta.defect_code != "D4hi":
                raise ValueError("LimitBacklashWrapper is only for explicit D4hi experiments")
            self._course_physics_dt = float(env.sim.model.opt.timestep)
            super().__init__(env, delta, seeds=seeds, family=family)

        def _xml_processor(self, xml):
            xml = rewrite_backlash_xml(super()._xml_processor(xml), self._plan,
                                       timestep=self._course_physics_dt)
            # The exact source used by the compiler avoids saveLastXML's float
            # rounding on robot transforms and mesh scales as well as limits.
            self._course_compiled_source_xml = xml
            return xml

        def reset(self, episode=None):
            result = super().reset(episode)
            configure_compiled_model(self.sim.model._model, self.sim.data._data,
                                     self.index.shadow_joint_ids, self.index.shadow_equality_ids)
            self._course_transmissions = tuple(Transmission.resolve(
                self.sim.model._model, s.parent_joint, s.name) for s in self.shadow_plan)
            self.state.engaged[:] = self.engagement_state()
            return result

        def engagement_state(self):
            # The parent collector calls sim.forward before taking a snapshot.
            # Do not run forward here: this read must not invalidate step1/step2.
            out = np.ones(self.index.n_joints, dtype=bool)
            for transmission in self._course_transmissions:
                i = self.index.joint_names.index(transmission.input_name)
                out[i] = transmission_diagnostics(self.sim.model._model,
                    self.sim.data._data, transmission)["transmitting"]
            return out

        def backlash_diagnostics(self):
            return [transmission_diagnostics(self.sim.model._model, self.sim.data._data, t)
                    for t in self._course_transmissions]

        def export_physics(self):
            xml = self._course_compiled_source_xml
            mapping = verify_exported_model(self.sim.model._model, xml)
            return {"model_xml": xml, "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                    "physics_version": PHYSICS_VERSION, "defect_state_contract": STATE_CONTRACT,
                    "state_map": mapping,
                    "transmissions": [t.__dict__ for t in self._course_transmissions],
                    "coordinate_rule": "output=input+relative; state=[qpos,qvel]",
                    "engagement_rule": "nonzero unilateral shadow-limit reaction",
                    "solver": {"timestep_s": self._course_physics_dt,
                               "solref": list(limit_solref(self._course_physics_dt)),
                               "solimp": list(LIMIT_SOLIMP)}}

    return LimitBacklashWrapper


def make_wrapper(env, delta, *, seeds, family):
    return backlash_wrapper_class()(env, delta, seeds=seeds, family=family)


def make_pair(task, family, level, seed_env):
    """Explicit alternative to collect_three_source.make_pair for new D4hi runs."""
    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import TwinPair
    from portal_trace.registry.tasks import task as task_spec
    from portal_trace.teacher import osc
    from portal_trace.types import SeedTriple
    spec = task_spec(task)
    if spec.tight_variant_of is not None:
        from portal_trace.registry.tight import ensure_registered
        ensure_registered(spec)
    bodies = []
    env = None
    try:
        env = osc.make_env(task, family=family, seed_env=seed_env)
        delta = Delta.from_level("D4hi", level,
            joint_scale=default_joint_scale("D4hi", len(env.robots[0].robot_joints)))
        seeds = SeedTriple(env=seed_env, train=-1, eval=0)
        bodies.append(make_wrapper(env, delta, seeds=seeds, family=family))
        env = osc.make_env(task, family=family, seed_env=seed_env)
        bodies.append(make_wrapper(env, delta.zero(), seeds=seeds, family=family))
        if model_state_map(bodies[0].sim.model._model) != model_state_map(bodies[1].sim.model._model):
            raise ValueError("clean and defective mechanical coordinates differ")
        return TwinPair(defective=bodies[0], clean=bodies[1], delta=delta,
                        family=family, task=task, extra_nodes=bodies[0].index.shadow_names)
    except BaseException:
        for body in bodies:
            body.close()
        if env is not None and (not bodies or env is not bodies[-1].env):
            env.close()
        raise


def save_physics_export(wrapper, output_prefix):
    """Write a reviewed XML/JSON pair; caller should store XML in each HDF5 too."""
    export = wrapper.export_physics()
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".xml").write_text(export.pop("model_xml"))
    prefix.with_suffix(".json").write_text(json.dumps(export, indent=2) + "\n")
    return export
