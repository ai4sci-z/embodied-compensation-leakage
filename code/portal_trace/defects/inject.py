"""Defect injection: one source XML, two compiled twins, and the 500 Hz hook.

This is the module 9.2 describes. It builds a structurally aligned twin pair,
wraps both in `DefectWrapper`, and implements all twelve catalogue entries
(D1, D2, D3a, D3b, D4hi, D4lo, D5, D6, D7, D8, D9, D10) across the three
injection layers of 9.2.0.

Where the writes land, and what that costs
------------------------------------------

`mjModel` writes (D8's `actuator_gainprm`, D4hi's `jnt_range` and
`eq_solref`/`eq_solimp`) are re-applied by `_apply_model_level` after **every**
reset. robosuite's `hard_reset` default is `True` (measured), and that branch
rebuilds `sim` from MJCF, so a write made once in the constructor disappears
mid-session with nothing in the log. `reset()` is the only entry point and it
always calls `_apply_model_level`; `env.reset` is rebound to it so that a caller
who reaches past the wrapper cannot skip the re-application.

`mjData` writes (`qfrc_applied` for D6 and D9) happen once per 500 Hz substep and
use whole-array assignment. `mj_step` does not zero `qfrc_applied` (9.1.1), so
`+=` would apply the same force 25 times per control step, and a single write at
the top of the control step would hold a stale sign across a reversal. The
assignment covers the whole array rather than the arm slice so that no residue
can survive anywhere in it.

Wrapper state has explicit reset semantics because `mj_step` manages none of it:

===========================  =================================================
state                        reset semantics
===========================  =================================================
D2 action buffer             filled with the zero-delta action, i.e. "hold"
                             (9.2.2 trap i); never with a stale command
D4lo slack `s`, direction    `s` set to the full band (engaged), direction
                             cleared, reference qpos taken from the reset pose
D10 drift accumulator        **not** reset -- it accumulates across episodes
                             within a session (9.2.9), and `new_session()` is
                             the only thing that clears it
D1/D10 sensing twin          reallocated, because `hard_reset` changes `nq`
===========================  =================================================

All of it is serialisable through `defect_state()` / `set_defect_state()`:
robomimic's flattened `states` cannot restore a frame for D2, D4 or D10
(9.2.0 discipline 3), so cross-replay writes both.

Those two are still not a whole frame, and the third column is robosuite's rather
than ours. `GripperModel.format_action` integrates the sign of the gripper
command instead of passing it through, so the finger setpoint is a saturating
integrator that lives in a Python attribute and not in `mjData` --
`actuation_state()` / `set_actuation_state()`, measured because a frame restored
without it misses `qvel` by up to 0.4 rad/s while the hand is on the object.

Branching discipline
--------------------

CONTRACT 3.3 forbids branching on the *dose*: there is no `if delta is None` and
no `if magnitude == 0` anywhere, so the clean twin runs the same arithmetic and a
floating-point bug in the injector shows up on both sides of the difference.
Branching on the *defect class* is unavoidable and is done through
`DefectSpec.primary_fields`, which is identical on the two sides of a twin pair
because `delta.zero()` preserves `defect_code`.

Bit-exact replay
----------------

Given (embodiment id, task, env seed) a replay reproduces byte for byte because:

1. `env.versions.check_pinned_versions` refuses to run on anything but
   mujoco 3.3.2 / robosuite 1.5.2, and MuJoCo promises nothing across versions.
2. Every source of randomness in robosuite 1.5.2 flows through `env.rng`
   (object placement, initial-joint noise, object choice -- verified by
   inspection), and `reset` re-derives it as
   `default_rng([env_seed, episode])` before the original reset runs. Both twins
   get the same stream, so a twin pair starts from the identical world state.
3. The wrapper's own state is a pure function of the action sequence and is
   dumped into `defect_state`. That plus `mj_setState` restores a frame exactly
   only together with `actuation_state()`: the gripper accumulator above is the
   one carrier neither of them covers.
4. The sensing twin mirrors robosuite's own accessors call for call --
   two separate `mj_jacSite` calls and `jac @ qvel`, not `mj_objectVelocity` --
   so at zero offset it returns the same bits rather than the same values.
5. `qfrc_applied` is assigned, never accumulated, and the model-level writes are
   recomputed from the `Delta` after each reset rather than restored from a
   snapshot taken before it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mujoco
import numpy as np
import robosuite
from robosuite import macros

from portal_trace.defects.delta import Delta
from portal_trace.defects.hooks import shadow_joint
from portal_trace.defects.params import WritableField, defect
from portal_trace.env.versions import check_pinned_versions
from portal_trace.registry.families import family as registry_family
from portal_trace.registry.tasks import (
    CONTROL_FREQ,
    HORIZON,
    SIMULATION_TIMESTEP,
    TaskSpec,
)
from portal_trace.types import SeedTriple

__all__ = [
    "D5Caliber",
    "ArmIndex",
    "ModelWriteRecord",
    "DefectWrapper",
    "TwinPair",
    "apply",
    "build_single",
    "build_twin_pair",
    "reseed_env",
    "gripper_command_state",
    "set_gripper_command_state",
    "sensing_offsets",
    "session_clock",
    "engagement_flags",
    "d5_caliber",
    "select_d5_caliber",
    "ZERO_VELOCITY_BAND",
    "D6_SYMMETRIC_FRACTION",
    "MAX_DELAY_STEPS",
]

# 9.2.6 trap ii: sign(qdot) chatters at zero, so the direction-dependent friction
# gets a dead band. Static friction is not modelled by it -- the qfrc_applied
# caliber gives that up on purpose, and P0 measures the offset (CONTRACT 3.5).
ZERO_VELOCITY_BAND = 1.0e-3

# 9.2.6: the symmetric base f_bar is fixed at 2% of the joint's rated torque.
D6_SYMMETRIC_FRACTION = 0.02

# 9.2.2 trap ii: the ring buffer has capacity n_max + 1 for every level, so all
# five levels share one memory layout and one code path.
MAX_DELAY_STEPS = 4


class D5Caliber(str, Enum):
    """9.2.5's two dead-zone calibers.

    `TOTAL_CTRL` is the primary: the dead zone is a property of the amplifier and
    the amplifier sees the whole torque command, gravity compensation included.
    `TASK_ONLY` applies it to the command net of gravity compensation, which is
    the variant 9.2.5 keeps in case the duty cycle on the total command is too
    low to be a dose at all.
    """

    TOTAL_CTRL = "total_ctrl"
    TASK_ONLY = "task_only"


_D5_CALIBER = D5Caliber.TOTAL_CTRL
_D5_CALIBER_USED: set[D5Caliber] = set()


def d5_caliber() -> D5Caliber:
    return _D5_CALIBER


def select_d5_caliber(caliber: D5Caliber) -> None:
    """Set the dead-zone caliber for the whole process.

    9.2.5 rule 3 makes this a single value across all six families: with two
    calibers in play the six D5 cells stop being six observations of one random
    effect and Proposition 1's two-way additive model fails on that column. The
    switch therefore refuses once an environment has been built on the other
    setting, rather than letting the two coexist.
    """
    global _D5_CALIBER
    caliber = D5Caliber(caliber)
    if _D5_CALIBER_USED and caliber not in _D5_CALIBER_USED:
        raise RuntimeError(
            f"D5 was already injected on the {sorted(c.value for c in _D5_CALIBER_USED)} "
            f"caliber in this process and {caliber.value} is being requested. 9.2.5 "
            "rule 3 requires one caliber across all six families; mixing them makes "
            "the six D5 cells six different physical quantities."
        )
    _D5_CALIBER = caliber


# ---------------------------------------------------------------------------
# Index resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmIndex:
    """Every index the injectors need, resolved by joint / actuator name.

    9.2.6 trap iii: a shadow joint shifts every dof index behind it, so nothing
    downstream may hold a bare number. This object is rebuilt after every reset.
    """

    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_index: tuple[int, ...]
    qvel_index: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    torque_limit: np.ndarray
    shadow_names: tuple[str, ...]
    shadow_joint_ids: tuple[int, ...]
    shadow_qpos_index: tuple[int, ...]
    shadow_qvel_index: tuple[int, ...]
    shadow_equality_ids: tuple[int, ...]
    ref_site_name: str
    ref_site_id: int
    armature: np.ndarray

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)


def _resolve_index(sim: Any, robot: Any, controller: Any, plan: tuple) -> ArmIndex:
    model = sim.model
    raw = model._model
    names = tuple(robot.robot_joints)
    joint_ids = tuple(int(model.joint_name2id(n)) for n in names)
    qpos_index = tuple(int(model.get_joint_qpos_addr(n)) for n in names)
    qvel_index = tuple(int(model.get_joint_qvel_addr(n)) for n in names)
    actuator_ids = tuple(int(i) for i in robot._ref_arm_joint_actuator_indexes)

    limited = raw.actuator_ctrllimited[list(actuator_ids)]
    hi = raw.actuator_ctrlrange[list(actuator_ids), 1]
    if not np.all(limited) or not np.all(hi > 0.0):
        raise ValueError(
            "D5 and D6 scale their dose by the joint's rated torque and this "
            f"family declares no positive ctrlrange on actuators {actuator_ids} "
            f"(ctrllimited={limited.tolist()}, high={hi.tolist()}). Panda ships "
            "forcerange='0 0' (unlimited) and ctrlrange +-80 / +-12, so the "
            "ctrlrange is the rated torque; a family without one has to declare "
            "it in the registry before D5 or D6 can be dosed on it."
        )

    shadow_names = tuple(spec.name for spec in plan)
    shadow_ids = tuple(int(model.joint_name2id(n)) for n in shadow_names)
    shadow_qpos = tuple(int(model.get_joint_qpos_addr(n)) for n in shadow_names)
    shadow_qvel = tuple(int(model.get_joint_qvel_addr(n)) for n in shadow_names)
    eq_ids = tuple(
        int(mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_EQUALITY, spec.equality_name))
        for spec in plan
        if spec.kind is shadow_joint.ShadowKind.BACKLASH
    )
    if any(i < 0 for i in eq_ids):
        raise KeyError("a backlash equality constraint is missing from the compiled model")

    return ArmIndex(
        joint_names=names,
        joint_ids=joint_ids,
        qpos_index=qpos_index,
        qvel_index=qvel_index,
        actuator_ids=actuator_ids,
        torque_limit=np.array(hi, dtype=np.float64),
        shadow_names=shadow_names,
        shadow_joint_ids=shadow_ids,
        shadow_qpos_index=shadow_qpos,
        shadow_qvel_index=shadow_qvel,
        shadow_equality_ids=eq_ids,
        ref_site_name=controller.ref_name,
        ref_site_id=int(model.site_name2id(controller.ref_name)),
        armature=np.array(raw.dof_armature[list(qvel_index)], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# The mjModel layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelWriteRecord:
    """Audit trail for 9.2.0 discipline 1, stored per trajectory."""

    fields: tuple[str, ...]
    gainprm_sha256: str
    frictionloss_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "gainprm_sha256": self.gainprm_sha256,
            "frictionloss_sha256": self.frictionloss_sha256,
        }


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def apply(
    model: Any,
    delta: Delta,
    *,
    index: ArmIndex,
    plan: tuple[shadow_joint.ShadowJointSpec, ...],
) -> ModelWriteRecord:
    """Write the mjModel-layer part of `delta` onto a freshly compiled model.

    `model` is a raw `mujoco.MjModel`. Evaluated at `delta.zero()` this produces
    the clean twin: the same assignments run with the identity parameters, so
    both sides of the pair take the same path through the same arithmetic
    (CONTRACT 3.3).
    """
    spec = defect(delta.defect_code)
    fields: list[str] = []

    if WritableField.ACTUATOR_GAINPRM in spec.primary_fields:
        gains = delta.joint_values("gain")
        for actuator_id, g in zip(index.actuator_ids, gains, strict=True):
            model.actuator_gainprm[actuator_id, 0] = g
        fields.append(WritableField.ACTUATOR_GAINPRM.value)

    if WritableField.JNT_RANGE in spec.primary_fields:
        written = False
        for joint_id, shadow in zip(index.shadow_joint_ids, plan, strict=True):
            if shadow.kind is not shadow_joint.ShadowKind.BACKLASH:
                continue
            model.jnt_limited[joint_id] = 1
            model.jnt_range[joint_id, 0] = -shadow.half_range
            model.jnt_range[joint_id, 1] = shadow.half_range
            written = True
        if written:
            fields.append(WritableField.JNT_RANGE.value)

    if WritableField.EQ_SOLREF_SOLIMP in spec.primary_fields and index.shadow_equality_ids:
        backlash = [s for s in plan if s.kind is shadow_joint.ShadowKind.BACKLASH]
        for eq_id, shadow in zip(index.shadow_equality_ids, backlash, strict=True):
            model.eq_solref[eq_id, :2] = shadow_joint.BACKLASH_SOLREF
            model.eq_solimp[eq_id, :5] = (
                *shadow_joint.BACKLASH_SOLIMP_HEAD,
                shadow.half_range,
                *shadow_joint.BACKLASH_SOLIMP_TAIL,
            )
        fields.append(WritableField.EQ_SOLREF_SOLIMP.value)

    return ModelWriteRecord(
        fields=tuple(fields),
        gainprm_sha256=_sha256(model.actuator_gainprm),
        frictionloss_sha256=_sha256(model.dof_frictionloss),
    )


# ---------------------------------------------------------------------------
# Wrapper-layer state
# ---------------------------------------------------------------------------


@dataclass
class WrapperState:
    """Everything `mj_step` does not manage, with its reset semantics attached."""

    action_buffer: np.ndarray            # (MAX_DELAY_STEPS + 1, action_dim)
    buffer_head: int
    effective_action: np.ndarray         # (action_dim,)
    play_slack: np.ndarray               # (n_joints,) travel since the last reversal
    play_direction: np.ndarray           # (n_joints,) -1 / 0 / +1
    play_reference_qpos: np.ndarray      # (n_joints,)
    engaged: np.ndarray                  # (n_joints,) bool, the memory state z of 5.1
    session_time: float                  # D10: seconds since the session started
    session_arc: float                   # D10: integral of ||qdot||, the motion variant

    def flat(self) -> np.ndarray:
        return np.concatenate(
            [
                self.action_buffer.reshape(-1),
                np.array([float(self.buffer_head)]),
                self.effective_action,
                self.play_slack,
                self.play_direction,
                self.play_reference_qpos,
                self.engaged.astype(np.float64),
                np.array([self.session_time, self.session_arc]),
            ]
        )

    def load(self, flat: np.ndarray) -> None:
        expected = self.flat().size
        if flat.size != expected:
            raise ValueError(f"defect_state has {flat.size} entries, this wrapper needs {expected}")
        n_buf = self.action_buffer.size
        n_a = self.effective_action.size
        n_j = self.play_slack.size
        cursor = 0

        def take(n: int) -> np.ndarray:
            nonlocal cursor
            out = flat[cursor : cursor + n]
            cursor += n
            return out

        self.action_buffer[:] = take(n_buf).reshape(self.action_buffer.shape)
        self.buffer_head = int(round(float(take(1)[0])))
        self.effective_action[:] = take(n_a)
        self.play_slack[:] = take(n_j)
        self.play_direction[:] = take(n_j)
        self.play_reference_qpos[:] = take(n_j)
        self.engaged[:] = take(n_j) > 0.5
        tail = take(2)
        self.session_time = float(tail[0])
        self.session_arc = float(tail[1])


def session_clock(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D10's two clocks out of flattened memory states: (N,) time and (N,) arc.

    `WrapperState.flat` writes them last, and this is the only reader outside
    `load` that depends on that. It exists so the off-branch query of 9.6.7 can
    recover the clock of every frame in a batch at once -- `load` would restore
    them one frame at a time and mutate the wrapper to do it, and the query is
    required not to advance the state it was asked about.
    """
    z = np.atleast_2d(np.asarray(flat, dtype=np.float64))
    return z[:, -2].copy(), z[:, -1].copy()


def engagement_flags(flat: np.ndarray, n_joints: int) -> np.ndarray:
    """Read the per-joint engagement flags from flattened defect state.

    D4lo writes these flags from its Python play operator.  D4hi writes the same
    stable slice from the shadow equality's live ``efc_pos`` read-out before a
    frame is flattened, so downstream persistence and analytic-field code need
    no caliber-specific state layout.
    """
    if n_joints < 1:
        raise ValueError(f"an engagement state needs at least one joint, got {n_joints}")
    z = np.atleast_2d(np.asarray(flat, dtype=np.float64))
    if z.shape[1] < n_joints + 2:
        raise ValueError(
            f"defect_state is {z.shape[1]} wide, too short for {n_joints} engagement "
            "flags followed by the two session clocks"
        )
    return (z[:, -(n_joints + 2) : -2] > 0.5).copy()


def sensing_offsets(
    delta: Delta, session_time: np.ndarray, session_arc: np.ndarray
) -> np.ndarray:
    """The perceived-encoder offset in radians: (N,) clocks -> (N, n_joints).

    9.2.1's D1 is a constant offset and 9.2.9's D10 is the same offset with a
    saturating drift law, dq(t) = dq_inf (1 - exp(-t / T)), run on session time
    or on the motion variant's arc length depending on which constant the level
    declares. Batched and free of wrapper state, because the ground truth of
    9.6.7 evaluates it at 45 M recorded frames whose clocks all differ, while
    the live wrapper needs exactly one frame -- `DefectWrapper.sensing_offset`
    is that one frame, so the law has a single implementation.
    """
    t = np.atleast_1d(np.asarray(session_time, dtype=np.float64))
    s = np.atleast_1d(np.asarray(session_arc, dtype=np.float64))
    if t.shape != s.shape:
        raise ValueError(f"session_time is {t.shape}, session_arc is {s.shape}")
    if "delta_q" in delta.params:
        return np.broadcast_to(
            delta.joint_values("delta_q"), (t.shape[0], delta.n_joints)
        ).copy()
    if "delta_q_inf" in delta.params:
        asymptote = delta.joint_values("delta_q_inf")
        if "tau_thermal" in delta.params:
            progress = 1.0 - np.exp(-t / float(delta.params["tau_thermal"]))
        else:
            progress = 1.0 - np.exp(-s / float(delta.params["s0_thermal"]))
        return progress[:, None] * asymptote[None, :]
    raise KeyError(
        f"{delta.defect_code} routes through the sensing twin but declares "
        "neither delta_q nor delta_q_inf"
    )


# ---------------------------------------------------------------------------
# Actuation memory that belongs to robosuite rather than to this wrapper
# ---------------------------------------------------------------------------


def _accumulating_grippers(env: Any) -> list[Any]:
    robot = env.robots[0]
    return [robot.gripper[arm] for arm in robot.arms if robot.has_gripper[arm]]


def gripper_command_state(env: Any) -> np.ndarray:
    """robosuite's gripper command accumulator, per arm, flattened.

    `GripperModel.format_action` does not pass the gripper action through: it
    integrates its sign at a fixed rate (0.2 per control step on the Panda hand)
    and clips to [-1, 1], so what the finger actuators receive is a saturating
    integrator of the command rather than the command. Had robosuite expressed
    that as a `dyntype="integrator"` actuator it would live in `mjData.act` and
    `mj_setState` would restore it; it is a Python attribute instead, so a frame
    restored from `mj_setState` plus `set_defect_state` is *not* restored -- the
    hand is left wherever the last control step drove it. During a grasp that is
    worth up to 0.4 rad/s of `qvel` after one control step (measured on
    Stack/Panda, table in `defects.adot.ClosedLoopDerivative`).

    Kept out of `defect_state` deliberately. That column is the memory state z of
    5.1, is owned by the injector, and is written to hdf5 at a width 9.6.3 fixes;
    this one is a property of the family's hand and exists on the clean twin too.
    """
    grippers = _accumulating_grippers(env)
    if not grippers:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(
        [np.asarray(g.current_action, dtype=np.float64).ravel() for g in grippers]
    )


def set_gripper_command_state(env: Any, flat: np.ndarray) -> None:
    """Write `gripper_command_state` back, refusing a width that cannot be split.

    The width is read off the live accumulators rather than off `gripper.dof`,
    because the two disagree: `Robot.reset` writes `zeros(gripper.dof)` -- one
    entry -- and `format_action` returns one entry per gripper actuator, two on
    the Panda hand, so the width changes on the first control step of an episode.
    A snapshot taken on one side of that boundary and restored on the other is a
    mistake rather than something to broadcast, which is what numpy would do.
    """
    flat = np.asarray(flat, dtype=np.float64).ravel()
    grippers = _accumulating_grippers(env)
    widths = [int(np.size(g.current_action)) for g in grippers]
    if flat.size != sum(widths):
        raise ValueError(
            f"gripper command state has {flat.size} entries and this body's "
            f"accumulators are {widths}. robosuite resets them to "
            "zeros(gripper.dof) and format_action then returns one entry per "
            "gripper actuator, so a value recorded mid-episode cannot be written "
            "onto a body that has only just been reset; step it once, or record "
            "the value after the reset settles"
        )
    cursor = 0
    for gripper, width in zip(grippers, widths, strict=True):
        gripper.current_action = flat[cursor : cursor + width].copy()
        cursor += width


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class DefectWrapper:
    """A robosuite env with one defect injected on all three layers.

    Constructed by `build_twin_pair`, which produces the defective and the clean
    member of a pair from the same source MJCF. The wrapper rebinds three of the
    env's methods (`_pre_action`, `reset`, and the arm controller's `update`) so
    that reaching past it is not a way to skip an injection step.
    """

    def __init__(self, env: Any, delta: Delta, *, seeds: SeedTriple, family: str) -> None:
        self.env = env
        self.delta = delta
        self.family = family
        self.seeds = seeds
        self.spec = defect(delta.defect_code)

        robot = env.robots[0]
        self.robot = robot
        if len(robot.arms) != 1:
            raise ValueError(
                f"{family} exposes arms {robot.arms}; the 9.2 injectors are written "
                "for the single-arm families of 9.3.1"
            )
        self.arm = robot.arms[0]

        if delta.n_joints != len(robot.robot_joints):
            raise ValueError(
                f"{delta} carries {delta.n_joints} joint scales and {family} has "
                f"{len(robot.robot_joints)} arm joints"
            )
        if delta.defect_code == "D5":
            _D5_CALIBER_USED.add(_D5_CALIBER)
        self.d5_caliber = _D5_CALIBER

        self._plan: tuple[shadow_joint.ShadowJointSpec, ...] = ()
        self._index: ArmIndex | None = None
        self._model_write: ModelWriteRecord | None = None
        self._frictionloss_reference: np.ndarray | None = None
        self._patched_controller: Any = None

        self._env_pre_action = env._pre_action
        self._env_reset = env.reset
        env._pre_action = self._pre_action
        env.reset = self.reset
        env.set_xml_processor(self._xml_processor)

        self.state = WrapperState(
            action_buffer=np.zeros((MAX_DELAY_STEPS + 1, env.action_dim)),
            buffer_head=0,
            effective_action=np.zeros(env.action_dim),
            play_slack=np.zeros(delta.n_joints),
            play_direction=np.zeros(delta.n_joints),
            play_reference_qpos=np.zeros(delta.n_joints),
            engaged=np.ones(delta.n_joints, dtype=bool),
            session_time=0.0,
            session_arc=0.0,
        )

        self._data_p: Any = None
        self._data_g: Any = None
        self._qfrc_buffer: np.ndarray | None = None
        self._jacp: np.ndarray | None = None
        self._jacr: np.ndarray | None = None
        self._tau_g: np.ndarray | None = None
        self._tau_g_stamp: float = np.nan
        # Captured at the first MuJoCo substep that applies each control-step
        # action.  This is the torque command at the same pre-step boundary as
        # the state snapshot, not qfrc_actuator from the following state.
        self._last_commanded_joint_torque: np.ndarray | None = None

        self._install_structure()

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped robosuite surface without bypassing ``step``/``reset``.

        The teacher APIs accept either a bare robosuite environment or this
        wrapper. Delegating reads such as ``sim``, ``robots`` and ``action_dim``
        keeps that contract real while the explicit methods below retain control
        over every state-changing operation.
        """
        return getattr(self.env, name)

    # -- structure -------------------------------------------------------

    def _xml_processor(self, xml: str) -> str:
        return shadow_joint.rewrite_xml(xml, self._plan)

    def _joint_geometry(self) -> tuple[dict, dict, dict]:
        model = self.env.sim.model
        raw = model._model
        axes, positions, armature = {}, {}, {}
        for name in self.robot.robot_joints:
            jid = int(model.joint_name2id(name))
            axes[name] = tuple(float(x) for x in raw.jnt_axis[jid])
            positions[name] = tuple(float(x) for x in raw.jnt_pos[jid])
            armature[name] = float(raw.dof_armature[int(raw.jnt_dofadr[jid])])
        return axes, positions, armature

    def _install_structure(self) -> None:
        """Compile the extra joints, in two passes when their inertia is needed.

        D3a's damping is 2 zeta sqrt(k I_eff) and D7's stiffness is proportional
        to I_eff; neither exists before the joint is in a compiled model, so the
        first pass compiles the node and the second binds the parameters.
        """
        axes, positions, armature = self._joint_geometry()
        self._plan = shadow_joint.plan_for(
            self.delta,
            arm_joint_names=tuple(self.robot.robot_joints),
            joint_axes=axes,
            joint_positions=positions,
            joint_armature=armature,
        )
        provisional = tuple(
            spec.with_inertia(1.0) if spec.needs_inertia else spec for spec in self._plan
        )
        self._plan = provisional
        self.reset()

        if any(spec.needs_inertia for spec in self._plan):
            inertia = self._shadow_inertia()
            self._plan = shadow_joint.resolve_inertia(
                shadow_joint.plan_for(
                    self.delta,
                    arm_joint_names=tuple(self.robot.robot_joints),
                    joint_axes=axes,
                    joint_positions=positions,
                    joint_armature=armature,
                ),
                inertia,
            )
            self.reset()

        if self._plan:
            self._assert_mass_matrix_regular()

    def _shadow_inertia(self) -> dict[str, float]:
        """Diagonal of the full mass matrix at the reset pose, per shadow dof.

        The shadow joints carry `armature = 0`, so this entry is the body-side
        inertia about the axis, which is the quantity the compliance mode swings.
        """
        raw = self.env.sim.model._model
        full = np.zeros((raw.nv, raw.nv))
        mujoco.mj_fullM(raw, full, self.env.sim.data.qM)
        idx = self.index.shadow_qvel_index
        return {
            name: float(full[i, i])
            for name, i in zip(self.index.shadow_names, idx, strict=True)
        }

    def _assert_mass_matrix_regular(self) -> None:
        """9.2.4 trap 3, checked instead of argued.

        Two co-located hinges are singular only if neither carries added inertia.
        Panda's driven joints do (armature 5 .. 0.714, measured), so the 2x2 block
        has determinant armature * I_body > 0 -- but that is a claim about a
        specific family, and the check is cheap.
        """
        raw = self.env.sim.model._model
        full = np.zeros((raw.nv, raw.nv))
        mujoco.mj_fullM(raw, full, self.env.sim.data.qM)
        arm = list(self.index.qvel_index) + list(self.index.shadow_qvel_index)
        block = full[np.ix_(arm, arm)]
        eigenvalues = np.linalg.eigvalsh(0.5 * (block + block.T))
        if eigenvalues.min() <= 0.0:
            raise ValueError(
                f"the arm-plus-shadow mass matrix has smallest eigenvalue "
                f"{eigenvalues.min():.3e} at the reset pose: the co-located joints "
                "can accelerate against each other with no inertia (9.2.4 trap 3). "
                "Give the driven joint a positive armature in the family MJCF and "
                "the identical value to the clean twin."
            )
        self.mass_matrix_condition = float(eigenvalues.max() / eigenvalues.min())

    # -- reset -----------------------------------------------------------

    def reset(self, episode: int | None = None) -> Any:
        """Reset, then re-apply everything `hard_reset` destroyed.

        Bound onto `env.reset` as well, so the model-level re-application is not
        something a caller can forget (9.2.0 discipline 1).
        """
        seed = [self.seeds.env] if episode is None else [self.seeds.env, int(episode)]
        reseed_env(self.env, seed)
        obs = self._env_reset()

        # `RobotEnv._load_model` runs `_load_robots`, which constructs a new
        # `Robot` and a new controller stack on every hard reset. Holding the
        # objects from construction would patch a robot the env no longer drives.
        self.robot = self.env.robots[0]
        self.arm = self.robot.arms[0]

        self._index = _resolve_index(self.env.sim, self.robot, self._controller(), self._plan)
        raw = self.env.sim.model._model
        if self._frictionloss_reference is None:
            self._frictionloss_reference = np.array(raw.dof_frictionloss, dtype=np.float64)
        self._model_write = apply(raw, self.delta, index=self._index, plan=self._plan)
        self._assert_frictionloss_untouched(raw)

        self._data_p = mujoco.MjData(raw)
        self._data_g = mujoco.MjData(raw)
        self._qfrc_buffer = np.zeros(raw.nv)
        self._jacp = np.zeros((3, raw.nv))
        self._jacr = np.zeros((3, raw.nv))
        self._tau_g = np.zeros(raw.nv)
        self._tau_g_stamp = np.nan

        self._reset_wrapper_state()
        self._install_sensing_patches()
        return obs

    def _assert_frictionloss_untouched(self, raw: Any) -> None:
        """CONTRACT 3.5, enforced rather than documented.

        D6 ships on the `qfrc_applied` caliber precisely so that
        `dof_frictionloss` stays at the family default on both sides of the pair
        (Panda: 0.1 N m on every arm joint -- measured, not zero). If it ever
        moves, `fgw_baseline` stops being zero and X1 goes with it.
        """
        if not np.array_equal(raw.dof_frictionloss, self._frictionloss_reference):
            raise AssertionError(
                "dof_frictionloss moved away from the family default. D6 ships on "
                "the qfrc_applied caliber (CONTRACT 3.5) exactly so that this array "
                "is bit-identical across the twin pair; a write here makes the "
                "baseline FGW distance non-zero."
            )

    def _reset_wrapper_state(self) -> None:
        st = self.state
        self._last_commanded_joint_torque = None
        # 9.2.2 trap i: prime with the zero-delta action, which is "hold pose"
        # under input_type="delta". Filling with zeros and treating them as an
        # absolute pose would inject an l-proportional transient at the start of
        # every episode and be read as a dose-response.
        st.action_buffer[:] = 0.0
        st.buffer_head = 0
        st.effective_action[:] = 0.0
        band = self.delta.joint_values("backlash") if "backlash" in self.delta.params else None
        st.play_slack[:] = band if band is not None else 0.0
        st.play_direction[:] = 0.0
        st.play_reference_qpos[:] = self.env.sim.data.qpos[list(self.index.qpos_index)]
        st.engaged[:] = True
        # session_time / session_arc deliberately survive: 9.2.9 requires D10's
        # drift to accumulate across episodes, otherwise it degenerates into a
        # per-episode repeat and stops breaking (A4), which is its only purpose.

    def new_session(self) -> None:
        """Clear D10's accumulator. The only thing that does (9.2.9)."""
        self.state.session_time = 0.0
        self.state.session_arc = 0.0

    # -- controller and observation patches --------------------------------

    def _controller(self) -> Any:
        return self.robot.part_controllers[self.arm]

    def _install_sensing_patches(self) -> None:
        """Route D1 / D10's encoder offset into the controller and the proprio obs.

        9.2.1's trap: adding the offset to `controller.joint_pos` alone leaves the
        end-effector loop exact, because OSC reads `site_xpos` from the *true*
        forward kinematics -- the defect cancels itself. Pose, orientation,
        Jacobian and task-space inertia all have to come from the sensing twin.
        Installed per defect class, not per dose, so the clean twin runs the same
        path.
        """
        if WritableField.SENSING_TWIN not in self.spec.primary_fields:
            return
        controller = self._controller()
        original_update = controller.update

        def biased_update(force: bool = False, _orig: Callable = original_update) -> None:
            # `Controller.update` is a no-op unless `new_update` is set, and the
            # gate has to be read before the call clears it. Rewriting the
            # perceived state on a call the controller itself skipped would give
            # the twin pair a refresh schedule the un-wrapped env does not have.
            refreshed = controller.new_update or force
            _orig(force=force)
            if refreshed:
                self._write_sensing_twin(controller)

        controller.update = biased_update
        self._patched_controller = controller
        # `update_initial_joints` stored the *commanded* home pose, and OSC's
        # nullspace term drives towards it; a biased encoder reports that pose
        # shifted. `initial_ref_pos` / `initial_ref_ori_mat` are left alone
        # because `update_initial_joints` binds them to the same arrays as
        # `ref_pos` / `ref_ori_mat`, which the sensing twin already writes
        # through -- rebinding them here would silently repair a robosuite
        # aliasing quirk on the defective side only.
        controller.initial_joint = (
            np.asarray(controller.initial_joint, dtype=np.float64) + self.sensing_offset()
        )
        # No twin write here. `update_initial_joints` leaves the cached pose and
        # mass matrix at the values from before the scene settled, and the first
        # `run_controller` of the episode consumes them as they stand; writing a
        # freshly computed state in would hand the pair a more current reading
        # than the un-wrapped env has. The bias enters at the first genuine
        # refresh, which is when the controller first reads the encoder.

        self._install_observable_patches()

    def _install_observable_patches(self) -> None:
        """Report the biased joint angles to the policy as well as the controller.

        9.2.1: if the proprio observation keeps the true angles while the
        controller sees biased ones, the policy input and the controller input
        disagree and the defect is no longer a self-consistent robot.
        """
        prefix = self.robot.robot_model.naming_prefix
        modality = f"{prefix}proprio"

        def sensed(_cache: dict) -> np.ndarray:
            return self._sensed_qpos()

        def sensed_cos(_cache: dict) -> np.ndarray:
            return np.cos(self._sensed_qpos())

        def sensed_sin(_cache: dict) -> np.ndarray:
            return np.sin(self._sensed_qpos())

        for fn in (sensed, sensed_cos, sensed_sin):
            fn.__modality__ = modality

        for name, fn in (
            (f"{prefix}joint_pos", sensed),
            (f"{prefix}joint_pos_cos", sensed_cos),
            (f"{prefix}joint_pos_sin", sensed_sin),
        ):
            if name in self.env._observables:
                self.env.modify_observable(observable_name=name, attribute="sensor", modifier=fn)

    # -- the sensing twin (D1, D10) ---------------------------------------

    def sensing_offset(self) -> np.ndarray:
        """The encoder offset in radians, per arm joint, at the live clock.

        D1 is constant; D10 is the same path with 9.2.9's drift law, whose
        accumulator runs on session time rather than episode time.
        """
        return sensing_offsets(self.delta, self.state.session_time, self.state.session_arc)[0]

    def _sensed_qpos(self) -> np.ndarray:
        return (
            np.asarray(self.env.sim.data.qpos[list(self.index.qpos_index)], dtype=np.float64)
            + self.sensing_offset()
        )

    def _write_sensing_twin(self, controller: Any) -> None:
        """Recompute the perceived kinematics and overwrite the controller's copy.

        Runs on every controller refresh, not once per control step as the 9.2.1
        listing suggests. robosuite calls `Controller.update()` inside
        `run_controller()` on every substep and it re-reads the true state, so a
        per-control-step refresh would be overwritten 24 times out of 25; holding
        the stale value instead would put a 50 ms zero-order hold inside the OSC
        feedback path, i.e. an in-loop delay, which is the very thing A-23 forbids
        conflating with D2.
        """
        sim = self.env.sim
        raw = sim.model._model
        data = sim.data._data
        dp = self._data_p
        idx = self.index

        dp.qpos[:] = data.qpos
        dp.qvel[:] = data.qvel
        dp.qpos[list(idx.qpos_index)] += self.sensing_offset()
        mujoco.mj_kinematics(raw, dp)
        mujoco.mj_comPos(raw, dp)
        mujoco.mj_crb(raw, dp)

        site = idx.ref_site_id
        # Two separate calls with one output each, mirroring robosuite's
        # get_site_jacp / get_site_jacr, so that a zero offset reproduces its
        # numbers bit for bit rather than to rounding.
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(raw, dp, self._jacp, None, site)
        mujoco.mj_jacSite(raw, dp, None, self._jacr, site)

        controller.joint_pos = np.array(dp.qpos[list(idx.qpos_index)])
        controller.ref_pos[:] = np.array(dp.site_xpos[site])
        controller.ref_ori_mat[:, :] = np.array(dp.site_xmat[site].reshape((3, 3)))
        controller.ref_pos_vel[:] = np.dot(self._jacp, dp.qvel)
        controller.ref_ori_vel[:] = np.dot(self._jacr, dp.qvel)
        controller.J_pos[:, :] = self._jacp[:, list(idx.qvel_index)]
        controller.J_ori[:, :] = self._jacr[:, list(idx.qvel_index)]
        controller.J_full[:, :] = np.vstack([controller.J_pos, controller.J_ori])

        # Allocated and sliced exactly as `Controller.update` does, down to the
        # bare `np.ndarray` and the two-stage fancy index. A reused buffer holds
        # the same bytes but lands at a different alignment, numpy dispatches
        # `inv` inside `opspace_matrices` to a different kernel, and the clean
        # twin picks up a 3e-15 drift against the un-wrapped env within three
        # control steps. Measured; the values were bit-identical throughout.
        mass = np.ndarray(shape=(raw.nv, raw.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(raw, mass, dp.qM)
        mass = np.reshape(mass, (raw.nv, raw.nv))
        qvel_index = list(idx.qvel_index)
        controller.mass_matrix = mass[qvel_index, :][:, qvel_index]

    # -- gravity torque (D3b, and D5's task_only caliber) ------------------

    def _gravity_torque(self) -> np.ndarray:
        """tau_g(q) on a shadow mjData with qdot = 0, per 9.2.3.

        Not `qfrc_bias`: that is gravity plus Coriolis plus centrifugal
        (verified), and scaling it would scale a velocity-dependent term too,
        which destroys the "purely static" property that makes D3b orthogonal to
        D2 and therefore usable as T8's carrier.
        """
        sim = self.env.sim
        raw = sim.model._model
        data = sim.data._data
        stamp = float(data.time)
        if stamp == self._tau_g_stamp:
            return self._tau_g
        dg = self._data_g
        dg.qpos[:] = data.qpos
        dg.qvel[:] = 0.0
        mujoco.mj_kinematics(raw, dg)
        mujoco.mj_comPos(raw, dg)
        mujoco.mj_comVel(raw, dg)
        mujoco.mj_rne(raw, dg, 0, self._tau_g)
        self._tau_g_stamp = stamp
        return self._tau_g

    def gravity_torque(self) -> np.ndarray:
        """Pure gravity torque on the arm joints at the current frame."""
        return np.asarray(
            self._gravity_torque()[list(self.index.qvel_index)], dtype=np.float64
        ).copy()

    # -- the 500 Hz hook ---------------------------------------------------

    def _pre_action(self, action: np.ndarray, policy_step: bool = False) -> None:
        """robosuite's injection point, called once per substep in both branches.

        `lite_physics=True` (the default, measured) puts this between `step1` and
        `step2`, which is the only correct cut: the reads (`qfrc_bias`,
        `site_xpos`, the Jacobians) come out of the position/velocity stage and
        the writes (`ctrl`, `qfrc_applied`) are consumed by the actuator and
        integration stage.
        """
        action = np.asarray(action, dtype=np.float64)
        if policy_step:
            self._advance_control_step(action)
        self._env_pre_action(self.state.effective_action, policy_step)
        self._rewrite_ctrl()
        if policy_step:
            self._last_commanded_joint_torque = self._joint_torque_from_ctrl()
        self._write_qfrc_applied()

    def _joint_torque_from_ctrl(self) -> np.ndarray:
        """Return the arm-joint torque command after the defect rewrite.

        The primary caliber rewrites ``data.ctrl`` before MuJoCo's actuator
        pipeline.  For the fixed-gain, bias-free motor actuators used by the
        registry, the joint torque is ``gear * gain * ctrl``.  Reading it here
        preserves the action/state boundary and avoids confusing the next
        state's ``qfrc_actuator`` with the command that generated this frame.
        """
        raw = self.env.sim.model._model
        ids = list(self.index.actuator_ids)
        gain = np.asarray(raw.actuator_gainprm[ids, 0], dtype=np.float64)
        gear = np.asarray(raw.actuator_gear[ids, 0], dtype=np.float64)
        if not np.allclose(raw.actuator_gear[ids, 1:], 0.0, atol=0.0, rtol=0.0):
            raise ValueError(
                "arm actuator gear has non-joint transmission terms; commanded "
                "joint torque is not identifiable from ctrl in this model"
            )
        return (gain * gear * np.asarray(self.env.sim.data.ctrl[ids], dtype=np.float64)).copy()

    def _advance_control_step(self, action: np.ndarray) -> None:
        """Per-control-step wrapper state. 9.2.2 trap iv: exactly once, not 25 times."""
        st = self.state
        # D2: the ring buffer runs for every defect, with n = 0 for the ones that
        # are not D2, so the delayed and undelayed paths are the same arithmetic.
        n = int(round(self._delay_steps()))
        cap = st.action_buffer.shape[0]
        st.action_buffer[st.buffer_head] = action
        st.effective_action[:] = st.action_buffer[(st.buffer_head - n) % cap]
        st.buffer_head = (st.buffer_head + 1) % cap

        dt = 1.0 / CONTROL_FREQ
        st.session_time += dt
        st.session_arc += float(
            np.linalg.norm(self.env.sim.data.qvel[list(self.index.qvel_index)]) * dt
        )

    def _delay_steps(self) -> float:
        if WritableField.CTRL_REWRITE not in self.spec.primary_fields:
            return 0.0
        if "n_control_steps" not in self.delta.params:
            return 0.0
        n = float(self.delta.params["n_control_steps"])
        if not 0.0 <= n <= MAX_DELAY_STEPS:
            raise ValueError(
                f"D2 level {self.delta.level} asks for {n} control steps of delay; "
                f"9.2.2 fixes the ladder at 0..{MAX_DELAY_STEPS} and the buffer at "
                "n_max + 1 so that every level shares one memory layout"
            )
        return n

    def _rewrite_ctrl(self) -> None:
        """D3b, D4lo and D5, in that order, then the amplifier clamp."""
        sim = self.env.sim
        raw = sim.model._model
        act = list(self.index.actuator_ids)
        u = np.asarray(sim.data.ctrl[act], dtype=np.float64)
        code = self.delta.defect_code

        if code == "D3b":
            sigma = self.delta.joint_values("sigma_g")
            u = u - sigma * self._gravity_torque()[list(self.index.qvel_index)]
        elif code == "D4lo":
            u = u * self._play_operator()
        elif code == "D5":
            width = self.delta.joint_values("w_over_tau_max") * self.index.torque_limit
            if self.d5_caliber is D5Caliber.TOTAL_CTRL:
                u = np.sign(u) * np.maximum(np.abs(u) - width, 0.0)
            else:
                tau_g = self._gravity_torque()[list(self.index.qvel_index)]
                task = u - tau_g
                u = np.sign(task) * np.maximum(np.abs(task) - width, 0.0) + tau_g

        lo = raw.actuator_ctrlrange[act, 0]
        hi = raw.actuator_ctrlrange[act, 1]
        sim.data.ctrl[act] = np.clip(u, lo, hi)

    def _play_operator(self) -> np.ndarray:
        """D4lo's hysteresis state machine: 1 where torque transmits, 0 in the slack.

        The memory state z of 5.1 in its low-fidelity form. It carries no
        transmission-side dynamics -- no free coasting inside the band, no impact
        on re-engagement -- which is exactly why 9.2.4 keeps it out of T8 and the
        safety endpoint and uses the shadow joint there instead.
        """
        st = self.state
        idx = list(self.index.qpos_index)
        qpos = np.asarray(self.env.sim.data.qpos[idx], dtype=np.float64)
        qvel = np.asarray(
            self.env.sim.data.qvel[list(self.index.qvel_index)], dtype=np.float64
        )
        band = self.delta.joint_values("backlash")

        direction = np.where(np.abs(qvel) > ZERO_VELOCITY_BAND, np.sign(qvel), 0.0)
        reversed_now = (
            (direction != 0.0) & (st.play_direction != 0.0) & (direction != st.play_direction)
        )
        st.play_slack[reversed_now] = 0.0
        st.play_slack += np.abs(qpos - st.play_reference_qpos)
        st.play_reference_qpos[:] = qpos
        st.play_direction[:] = np.where(direction != 0.0, direction, st.play_direction)
        st.engaged[:] = st.play_slack >= band
        return st.engaged.astype(np.float64)

    def _write_qfrc_applied(self) -> None:
        """D6 and D9. Whole-array assignment, every substep (9.1.1, 9.2.0 rule 2)."""
        buf = self._qfrc_buffer
        buf[:] = 0.0
        dof = list(self.index.qvel_index)
        code = self.delta.defect_code

        if code == "D6":
            qvel = np.asarray(self.env.sim.data.qvel[dof], dtype=np.float64)
            sgn = np.where(np.abs(qvel) > ZERO_VELOCITY_BAND, np.sign(qvel), 0.0)
            # CONTRACT 3.5: the symmetric base and the directional half both go
            # through this one write, so dof_frictionloss keeps the family default.
            f_bar = D6_SYMMETRIC_FRACTION * self.index.torque_limit
            half_delta = 0.5 * self.delta.joint_values("delta_f")
            buf[dof] = -(f_bar + half_delta * sgn) * sgn
        elif code == "D9":
            qpos = np.asarray(self.env.sim.data.qpos[list(self.index.qpos_index)], dtype=np.float64)
            amplitude = self.delta.joint_values("amplitude")
            poles = self.delta.joint_values("n_poles")
            phase = self.delta.joint_values("phase")
            buf[dof] = -amplitude * np.sin(poles * qpos + phase)

        self.env.sim.data.qfrc_applied[:] = buf

    # -- pass-through and state I/O ---------------------------------------

    @property
    def index(self) -> ArmIndex:
        if self._index is None:
            raise RuntimeError("reset() has not run; there are no resolved indices yet")
        return self._index

    @property
    def model_write(self) -> ModelWriteRecord:
        if self._model_write is None:
            raise RuntimeError("reset() has not run; nothing has been written to mjModel yet")
        return self._model_write

    @property
    def shadow_plan(self) -> tuple[shadow_joint.ShadowJointSpec, ...]:
        return self._plan

    def step(self, action: np.ndarray) -> Any:
        return self.env.step(action)

    def defect_state(self) -> np.ndarray:
        """The 9.2.0 discipline 3 companion to robomimic's flattened `states`.

        D4hi's engagement is physical simulator state, not the Python play
        operator used by D4lo.  Refresh the shared engagement slice from the
        equality constraint immediately before every snapshot so the existing
        collection path persists the state that actually generated this frame.
        """
        if self.delta.defect_code == "D4hi":
            self.state.engaged[:] = self.engagement_state()
        return self.state.flat()

    def engagement_state(self) -> np.ndarray:
        """Current per-joint engagement, from the owning caliber's state."""
        if self.delta.defect_code != "D4hi":
            return self.state.engaged.copy()

        backlash = tuple(
            spec for spec in self._plan if spec.kind is shadow_joint.ShadowKind.BACKLASH
        )
        equality_ids = self.index.shadow_equality_ids
        if len(backlash) != len(equality_ids):
            raise RuntimeError(
                f"D4hi has {len(backlash)} backlash shadow joints but "
                f"{len(equality_ids)} equality ids"
            )
        observed = shadow_joint.backlash_engagement(
            self.env.sim.data._data,
            equality_ids,
            tuple(float(spec.half_range) for spec in backlash),
        )
        out = np.ones(self.index.n_joints, dtype=bool)
        by_name = {name: i for i, name in enumerate(self.index.joint_names)}
        for spec, value in zip(backlash, observed, strict=True):
            try:
                joint_index = by_name[spec.parent_joint]
            except KeyError as exc:
                raise RuntimeError(
                    f"D4hi shadow {spec.name!r} names unknown arm joint "
                    f"{spec.parent_joint!r}"
                ) from exc
            out[joint_index] = bool(value)
        return out

    def set_defect_state(self, flat: np.ndarray) -> None:
        self.state.load(np.asarray(flat, dtype=np.float64))

    def synchronise_episode_start(self) -> None:
        """Reset episode-local memory after a twin state has been synchronised.

        Settling is scene preparation rather than part of a defective episode.
        A paired reset may therefore copy the clean settled physics state onto
        this body before frame zero. Reinitialise the delay/play state against
        that copied state while preserving D10's session clock, exactly as a
        normal episode reset does.
        """
        self._reset_wrapper_state()

    def actuation_state(self) -> np.ndarray:
        """The third column a frame needs: robosuite's gripper accumulator.

        Separate from `defect_state` because it is not the injector's -- it is
        present at delta = 0 and is the same trace on both members of a twin
        pair, being driven by the sign of the gripper command alone.
        """
        return gripper_command_state(self.env)

    def commanded_joint_torque(self) -> np.ndarray:
        """Torque command captured for the current control-step action."""
        if self._last_commanded_joint_torque is None:
            raise RuntimeError(
                "commanded joint torque is unavailable before the first control "
                "step; do not fabricate a frame-zero value"
            )
        return self._last_commanded_joint_torque.copy()

    def set_actuation_state(self, flat: np.ndarray) -> None:
        set_gripper_command_state(self.env, flat)

    def close(self) -> None:
        self.env.close()


@dataclass(frozen=True)
class TwinPair:
    """A structurally aligned pair, both members built by the same call."""

    defective: DefectWrapper
    clean: DefectWrapper
    delta: Delta
    family: str
    task: str
    extra_nodes: tuple[str, ...] = field(default_factory=tuple)

    def __iter__(self):
        return iter((self.defective, self.clean))


def reseed_env(env: Any, seed: list[int]) -> None:
    """Put the env and every object sampler on one freshly seeded generator.

    `MujocoEnv.rng` is not the only stream that decides an episode.
    `NutAssembly._load_model` builds its `SequentialCompositeSampler` only when
    `self.placement_initializer is None`, so the sampler is constructed once, on
    the first load, holding a reference to whatever generator `self.rng` was then
    -- every later hard reset reuses it and continues its stream. Assigning
    `env.rng` alone therefore controls object placement on the first episode and
    on no other, and two envs that have been reset a different number of times
    diverge in the nut pose while agreeing everywhere else. Rebinding the sampler
    is what makes (embodiment, task, env seed) actually determine the episode.
    """
    generator = np.random.default_rng(seed)
    env.rng = generator
    sampler = getattr(env, "placement_initializer", None)
    if sampler is None:
        return
    pending = [sampler]
    while pending:
        node = pending.pop()
        node.rng = generator
        pending.extend(getattr(node, "samplers", {}).values())


def _robosuite_robot(family_name: str) -> str:
    """The registry's resolution of a 9.3.1 family, not a second mapping."""
    name = registry_family(family_name).robosuite_robot
    if name not in robosuite.ALL_ROBOTS:
        raise KeyError(
            f"family {family_name!r} resolves to robot {name!r}, which this "
            "robosuite install does not register. 9.3.1 sources four of the ten "
            "families from mujoco_menagerie or robosuite_models; those have to be "
            f"installed before they can be built. Known: {sorted(robosuite.ALL_ROBOTS)}"
        )
    return name


def _make_env(
    task_spec: TaskSpec,
    family: str,
    seeds: SeedTriple,
    *,
    reward_shaping: bool | None = None,
) -> Any:
    if macros.SIMULATION_TIMESTEP != SIMULATION_TIMESTEP:
        raise ValueError(
            f"robosuite macros.SIMULATION_TIMESTEP is {macros.SIMULATION_TIMESTEP}, "
            f"9.1.1 locks it at {SIMULATION_TIMESTEP}; the 500 Hz injection hook and "
            "the whole budget chapter are computed from it"
        )
    kwargs = dict(task_spec.env_kwargs)
    kwargs.update(
        robots=_robosuite_robot(family),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        lite_physics=True,
        hard_reset=True,
        ignore_done=False,
        seed=seeds.env,
    )
    if reward_shaping is not None:
        # `None` means "leave the task class's own default alone", which is not
        # the same as False: robosuite's shaping default is per task (`Wipe`
        # ships True, the rest False), so writing a literal False here would
        # change one registry task while claiming to change nothing. Every
        # demonstration path leaves it unset, because the ILC teacher tracks a
        # reference trajectory and never reads the task reward. The one caller
        # that sets it is A13's RL level, whose agent optimises this reward and
        # therefore has to record which one it was.
        kwargs["reward_shaping"] = reward_shaping
    if task_spec.tight_variant_of is not None:
        # Tight-fit tasks are project-defined subclasses, not robosuite natives.
        # Register the subclass explicitly before calling suite.make; relying on
        # import side effects would make the formal runtime matrix fail by task.
        from portal_trace.registry.tight import ensure_registered

        ensure_registered(task_spec)
    env = robosuite.make(task_spec.env_name, **kwargs)
    if env.sim.model._model.opt.noslip_iterations != 0:
        raise ValueError(
            "noslip_iterations is non-zero. 9.2.0 discipline 5: the NoSlip "
            "post-solver drives qfrc_constraint to near zero on dofs with "
            "frictionloss and breaks equality/joint coupling, which is exactly the "
            "combination D6 and D4hi produce."
        )
    return env


def build_single(
    task_spec: TaskSpec,
    family: str,
    delta: Delta,
    seeds: SeedTriple,
    *,
    reward_shaping: bool | None = None,
) -> DefectWrapper:
    """One side of a pair, built without reference to the other.

    Experiments never want this: 9.3.3 compares a defective robot to its clean
    twin, and building one side alone invites comparing it to something built by
    a different code path. Two things do. W1 calibration, where the identity
    value that `delta.zero()` would need is the very quantity being measured --
    D7's rigid-side compliance is capped by the integrator, so it has to be swept
    before `Delta.zero()` on D7 can answer at all. And A13's RL level
    (`ablations.teacher_family_rl`), whose expert is trained on the defective
    embodiment alone: 10.6.1 asks what an agent arrives at on the broken robot,
    so a clean twin has no role in that run and building one would double its
    cost.
    """
    check_pinned_versions(required=("mujoco", "robosuite"))
    return DefectWrapper(
        _make_env(task_spec, family, seeds, reward_shaping=reward_shaping),
        delta,
        seeds=seeds,
        family=family,
    )


def build_twin_pair(
    task_spec: TaskSpec,
    family: str,
    delta: Delta,
    seeds: SeedTriple,
) -> TwinPair:
    """The defective robot and its clean twin, from one source MJCF.

    Both members are `DefectWrapper(env, .)`; the clean one takes
    `delta.zero()`, so it is the same `apply` running with the identity
    parameters rather than a separate class or an early return (CONTRACT 3.3,
    9.2.0 discipline 4). For the graph-changing defects that also means the clean
    twin compiles the same extra node -- D4hi's shadow joint with the +-10^-5 rad
    band of 9.3.4 -- at the identical position and axis.
    """
    defective = build_single(task_spec, family, delta, seeds)
    clean = build_single(task_spec, family, delta.zero(), seeds)
    if defective.env.sim.model._model.nv != clean.env.sim.model._model.nv:
        raise AssertionError(
            "the twin pair compiled to different dof counts; the clean twin has to "
            "carry the same extra joints (9.3.4)"
        )
    return TwinPair(
        defective=defective,
        clean=clean,
        delta=delta,
        family=family,
        task=task_spec.task_id,
        extra_nodes=defective.index.shadow_names,
    )
