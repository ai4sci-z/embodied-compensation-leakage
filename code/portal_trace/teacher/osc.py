"""The deterministic OSC tracker, with gains frozen once (9.4.2, 9.4.6).

The tracker is robosuite's own `OSC_POSE` composite controller at the values
9.1.1 locks: `kp=150`, `damping_ratio=1`, `impedance_mode="fixed"`,
`input_type="delta"`, `input_ref_frame="base"`,
`output_max=[0.05,0.05,0.05,0.5,0.5,0.5]`, `uncouple_pos_ori=true`,
`interpolation=null`. What this module adds on top is the layer that turns an
absolute task-space reference pose into the delta action that OSC consumes, plus
the bookkeeping that makes a rollout reproducible.

**Gains are a property of the tracker, not an argument.** 9.4.6 forbids
per-embodiment tuning: a gain retuned on one body is a hidden compensation
source that enters neither delta nor the ground truth of c_delta, and it moves
every Lambda in a way nothing can attribute. `OSCTracker` therefore has no gain
parameter at all -- `OSCTracker.gains` returns the module-level `FROZEN_GAINS`
and there is no setter. The 9.4.6 grid over `kp in {100,150,200,250}` is
implemented as `tune_gains`, an audit that re-derives the frozen value on the
clean twins and reports disagreement; it cannot change what a rollout uses.

**Determinism.** 9.1.6 requires that (embodiment, task, `seed_env`) reproduce a
trajectory bit for bit. Three things make that hold here: `seed_env` is pushed
into the global numpy RNG that robosuite's placement sampler draws from, before
`reset`; the tracker itself is a pure function of the measured pose and the
reference, with no internal state that survives a rollout; and every rollout
records the flattened MuJoCo state at each control step, so `replay` can restore
any frame exactly. `assert_deterministic` checks the property rather than
asserting it in prose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation

from portal_trace.defects.inject import gripper_command_state
from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON, TaskSpec
from portal_trace.registry.tasks import task as task_spec
from portal_trace.success import terminal_hold_start
from portal_trace.teacher.refs import ReferenceTrajectory, load_spec, resolve

__all__ = [
    "OSCGains",
    "FROZEN_GAINS",
    "FROZEN_KP",
    "KP_GRID",
    "ACTION_DIM",
    "env_action",
    "physics_state",
    "set_physics_state",
    "OSCTracker",
    "TrackingResult",
    "GainTuningReport",
    "make_env",
    "reset_env",
    "reset_twin_pair",
    "tune_gains",
    "assert_deterministic",
    "mating_geom_ids",
    "wrist_ft",
    "mode_code",
    "mode_sequence",
    "fold_mode_codes",
]

# 9.1.1 / 9.4.2. Stated in the proposal, therefore a constant here and not a
# lock-file measurement (contract 3.1). 9.4.6 additionally requires that the
# value be re-derivable by the kp grid below; `tune_gains` is that derivation.
FROZEN_KP = 150.0
KP_GRID: tuple[float, ...] = (100.0, 150.0, 200.0, 250.0)

# 9.6.1: 6 delta-pose channels plus one gripper channel, all in [-1, 1].
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# robosuite's placement sampler drops objects from a standoff above the table:
# a square nut is released at z = 0.89 and comes to rest at 0.83, about ten
# control steps later. The reference resolver reads object poses once, at t = 0,
# so resolving against the un-settled state would hang every waypoint 6 cm above
# the object. Settling is therefore part of the reset, before the horizon starts,
# and is held long enough that the pose the resolver reads is the pose the
# demonstration will actually find. Not a measurement: a collection parameter.
SETTLE_STEPS = 20
_HOLD_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def env_action(env: Any, action: np.ndarray) -> np.ndarray:
    """Narrow a canonical 9.6.1 action to what this environment's actuators accept.

    Every episode is recorded in the 7-dim canonical space so that one action
    column serves every cell, but Wipe mounts a gripper with no actuated joint
    and its environment takes 6. Dropping the gripper channel is the whole of the
    difference; anything else is a mismatch worth failing on rather than padding
    over, because a silently reshaped action would be recorded as if it had been
    applied.
    """
    n = int(env.action_dim)
    if n == ACTION_DIM:
        return action
    if n == ACTION_DIM - 1 and env.robots[0].gripper["right"].dof == 0:
        return action[..., :n]
    raise ValueError(
        f"environment takes {n} action dimensions and the canonical space is "
        f"{ACTION_DIM}; only a gripper with no actuated joint may account for the "
        "difference"
    )


def _joint_actuator_ids(env: Any) -> tuple[int, ...]:
    """Resolve arm actuator ids for a wrapper or a bare robosuite env."""
    if callable(getattr(env, "commanded_joint_torque", None)):
        return tuple(range(int(env.index.n_joints)))
    robot = env.robots[0]
    return tuple(int(i) for i in robot._ref_arm_joint_actuator_indexes)


def _commanded_joint_torque(env: Any, actuator_ids: tuple[int, ...]) -> np.ndarray:
    """Read the fixed-gain motor command at the current control boundary."""
    if callable(getattr(env, "commanded_joint_torque", None)):
        return np.asarray(env.commanded_joint_torque(), dtype=np.float64)
    raw = env.sim.model._model
    ids = list(actuator_ids)
    if int(raw.nu) <= max(ids, default=-1):
        raise ValueError("arm actuator id is outside the compiled MuJoCo model")
    if not np.allclose(raw.actuator_gear[ids, 1:], 0.0, atol=0.0, rtol=0.0):
        raise ValueError(
            "arm actuator gear has non-joint transmission terms; commanded joint "
            "torque is not identifiable from ctrl in this model"
        )
    if not np.all(raw.actuator_gaintype[ids] == 0) or not np.all(raw.actuator_biastype[ids] == 0):
        raise ValueError(
            "arm actuator gain/bias is not fixed and bias-free; commanded joint "
            "torque is not identifiable from ctrl in this model"
        )
    gain = np.asarray(raw.actuator_gainprm[ids, 0], dtype=np.float64)
    gear = np.asarray(raw.actuator_gear[ids, 0], dtype=np.float64)
    return (gain * gear * np.asarray(env.sim.data.ctrl[ids], dtype=np.float64)).copy()


@dataclass(frozen=True)
class OSCGains:
    """The arm controller block of the composite config.

    Frozen dataclass with a content hash, because 9.4.6 requires
    `controller_config_sha256` to travel with every trajectory: that hash is what
    makes "the gains were never retuned" checkable after the fact instead of
    promised.
    """

    kp: float = FROZEN_KP
    damping_ratio: float = 1.0
    impedance_mode: str = "fixed"
    input_max: float = 1.0
    input_min: float = -1.0
    output_max: tuple[float, ...] = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    output_min: tuple[float, ...] = (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5)
    kp_limits: tuple[float, float] = (0.0, 300.0)
    damping_ratio_limits: tuple[float, float] = (0.0, 10.0)
    input_type: str = "delta"
    input_ref_frame: str = "base"
    uncouple_pos_ori: bool = True
    ramp_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.impedance_mode != "fixed":
            raise ValueError(
                "9.4.2 fixes impedance_mode to 'fixed'. The variable modes put kp "
                "into the action space, which would make the gain a per-frame "
                "quantity learned from data -- exactly the hidden compensation "
                "source 9.4.6 rules out"
            )
        lo, hi = self.kp_limits
        if not (lo <= self.kp <= hi):
            raise ValueError(f"kp={self.kp} outside kp_limits {self.kp_limits}")

    @property
    def pos_scale(self) -> np.ndarray:
        """Metres per action unit on the three translation channels."""
        return np.asarray(self.output_max[:3], dtype=np.float64)

    @property
    def ori_scale(self) -> np.ndarray:
        """Radians per action unit on the three rotation channels."""
        return np.asarray(self.output_max[3:6], dtype=np.float64)

    def to_arm_config(self) -> dict[str, Any]:
        return {
            "type": "OSC_POSE",
            "input_max": self.input_max,
            "input_min": self.input_min,
            "output_max": list(self.output_max),
            "output_min": list(self.output_min),
            "kp": self.kp,
            "damping_ratio": self.damping_ratio,
            "impedance_mode": self.impedance_mode,
            "kp_limits": list(self.kp_limits),
            "damping_ratio_limits": list(self.damping_ratio_limits),
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": self.uncouple_pos_ori,
            "input_type": self.input_type,
            "input_ref_frame": self.input_ref_frame,
            "interpolation": None,
            "ramp_ratio": self.ramp_ratio,
            "gripper": {"type": "GRIP"},
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_arm_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


FROZEN_GAINS = OSCGains()


def _composite_config(family: str, gains: OSCGains) -> dict[str, Any]:
    cfg = load_composite_controller_config(controller="BASIC", robot=family)
    body_parts = cfg["body_parts"]
    arm = gains.to_arm_config()
    for part in ("right", "left"):
        if part in body_parts:
            body_parts[part] = dict(arm)
    return cfg


def make_env(
    task_id: str,
    *,
    family: str,
    seed_env: int,
    gains: OSCGains = FROZEN_GAINS,
    hard_reset: bool = True,
    lite_physics: bool = True,
    render_offscreen: bool = False,
    camera_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
) -> Any:
    """Build the robosuite environment for one (task, family) cell.

    Rendering is off by default: 9.1.3 requires the ILC iteration stage to run
    without pixels, and 9.4.2 puts the whole cost argument for 20 iterations on
    that. `render_offscreen=True` is for the final collection rollout only.

    `gains` is here so that `tune_gains` can build the grid. Every production
    caller leaves it at `FROZEN_GAINS`; `OSCTracker` never passes anything else.
    """
    spec: TaskSpec = task_spec(task_id)
    kwargs = dict(spec.env_kwargs)
    kwargs.pop("horizon", None)
    env = suite.make(
        spec.env_name,
        robots=family,
        controller_configs=_composite_config(family, gains),
        has_renderer=False,
        has_offscreen_renderer=render_offscreen,
        use_camera_obs=render_offscreen,
        camera_names=list(camera_names) if render_offscreen else None,
        camera_heights=84,
        camera_widths=84,
        reward_shaping=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        ignore_done=True,
        hard_reset=hard_reset,
        lite_physics=lite_physics,
        seed=int(seed_env) % (2 ** 32),
        **kwargs,
    )
    reset_env(env, seed_env)
    return env


def reset_env(env: Any, seed_env: int, *, settle_steps: int = SETTLE_STEPS) -> np.ndarray:
    """Reset to the initial state `seed_env` selects, and return the flat state.

    Reseeding is done by writing the bit generator state of `env.rng` in place,
    not by assigning a new generator. robosuite builds its `ObjectPositionSampler`
    once, in `_load_model`, holding a reference to the generator object; replacing
    `env.rng` would leave the sampler drawing from the old one, and object
    placement -- the thing `seed_env` is supposed to fix -- would stay free. The
    legacy global RNG is seeded alongside it because parts of the object suite
    still reach for `np.random` directly.

    Because settling through a live defect can itself move the body, callers
    that need a paired experimental start use `reset_twin_pair`, which copies
    the clean settled state and actuation onto the defective member before frame
    zero. `reset_env` remains the single-body primitive used underneath it.
    """
    seed = int(seed_env) % (2 ** 32)
    env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    np.random.seed(seed)
    env.reset()
    # A zero delta action in "achieved" goal mode means "hold the pose you are
    # in", so the arm stays put while the scene settles.
    hold = env_action(env, _HOLD_ACTION)
    for _ in range(settle_steps):
        env.step(hold)
    # Attribute reads on DefectWrapper delegate to its robosuite env, but
    # assignments do not. Writing ``env.timestep`` on the wrapper would leave
    # the real episode clock at ``settle_steps`` and terminate a nominal
    # 400-frame rollout after only 390 policy steps.
    episode_owner = getattr(env, "env", env)
    episode_owner.timestep = 0
    episode_owner.done = False
    return physics_state(env.sim)


def reset_twin_pair(pair: Any, seed_env: int) -> np.ndarray:
    """Reset a twin pair onto one clean settled state before frame zero.

    Defects must not change the initial-state distribution. Running settling
    through each live wrapper independently violates that for delay, sensing,
    torque and graph-level interventions. Both members are still reset and
    settled so their controller and gripper internals cross the same lifecycle;
    the clean member's resulting physics and actuation state then becomes the
    shared episode start, and episode-local wrapper memory is reinitialised
    against it.
    """
    reset_env(pair.defective, seed_env)
    clean_state = reset_env(pair.clean, seed_env)
    clean_actuation = pair.clean.actuation_state()
    set_physics_state(pair.defective.sim, clean_state)
    pair.defective.set_actuation_state(clean_actuation)
    for side in pair:
        side.synchronise_episode_start()
    defective_state = physics_state(pair.defective.sim)
    if not np.array_equal(defective_state, clean_state):
        raise AssertionError(
            f"{pair.task}/{pair.family}/{pair.delta.defect_code}: twin states are "
            f"not bit-identical after paired reset at environment seed {seed_env}"
        )
    if not np.array_equal(pair.defective.actuation_state(), clean_actuation):
        raise AssertionError(
            f"{pair.task}/{pair.family}/{pair.delta.defect_code}: twin gripper "
            f"accumulators differ after paired reset at environment seed {seed_env}"
        )
    return clean_state.copy()


def physics_state(sim: Any) -> np.ndarray:
    """The (nq + nv,) state vector every module in the package passes around.

    `Episode.states` and `CompensationField.evaluate` both fix this width, so the
    tracker records it rather than robosuite's `get_state().flatten()`, which
    prepends `data.time`. One convention, because the off-branch query of 9.6.7
    moves a recorded state between two models and a silent one-column offset
    there would land as a plausible-looking wrong action rather than as an error.
    """
    data = sim.data
    return np.concatenate(
        [np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)]
    )


def set_physics_state(sim: Any, state: np.ndarray) -> None:
    """Write a (nq + nv,) state back and re-run forward kinematics.

    The counterpart of `physics_state`, and the move 9.6.7 item 1 calls bit-exact:
    a twin pair compiles to the same `nq`/`nv`, so a state recorded on one member
    is written to the other without any remapping.
    """
    state = np.asarray(state, dtype=np.float64)
    nq, nv = int(sim.model.nq), int(sim.model.nv)
    if state.size != nq + nv:
        raise ValueError(f"state has {state.size} entries, this model needs {nq + nv}")
    sim.data.qpos[:] = state[:nq]
    sim.data.qvel[:] = state[nq:]
    sim.forward()


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------


def _rotvec(mats: np.ndarray) -> np.ndarray:
    """Rotation vectors of a stack of rotation matrices, in radians."""
    single = mats.ndim == 2
    r = Rotation.from_matrix(mats.reshape(-1, 3, 3))
    out = r.as_rotvec()
    return out[0] if single else out


@dataclass
class TrackingResult:
    """One rollout of the tracker along X_tau.

    `states` is what `mj_setState` needs for bit-exact restoration (9.1.6 item 4)
    and is the column `replay` and `query` both key on. `error` is in metres and
    radians, never in action units: 9.4.2 is explicit that the ILC error signal
    is a physical quantity, because expressing it in action units would let
    `output_max` clipping contaminate it.

    `actuation` is next to `states` because `states` is not a whole frame:
    robosuite's gripper command accumulator is a saturating integrator that lives
    outside `mjData`, and a one-step push-forward from `states` alone misses
    `qvel` by up to 0.40 rad/s during a grasp (`defects.adot`). It is recorded at
    the same instant as `states[t]`, i.e. before the step, so the two restore the
    same frame.

    `tau_g` and `defect_state` share that pre-step boundary. Bare robosuite
    environments expose empty arrays for the wrapper-only columns; collection
    refuses those arrays rather than storing an unreplayable demonstration.
    """

    task_id: str
    family: str
    seed_env: int
    states: np.ndarray            # (T, nq + nv), the Episode.states convention
    actuation: np.ndarray         # (T, k) robosuite's gripper command accumulator
    actions: np.ndarray           # (T, 7) clipped, as the wrapper received them
    actions_raw: np.ndarray       # (T, 7) before the action-range clip
    eef_pos_cmd: np.ndarray       # (T, 3) pose the action was computed from
    eef_rot_cmd: np.ndarray       # (T, 3, 3)
    eef_pos_ach: np.ndarray       # (T, 3) pose realised at the end of the step
    eef_rot_ach: np.ndarray       # (T, 3, 3)
    error: np.ndarray             # (T, 6) [m, m, m, rad, rad, rad]
    wrist_ft: np.ndarray          # (T, 6) force/torque at the eef sensor
    tau_g: np.ndarray             # (T, J) pure gravity torque, empty for a bare env
    joint_torque: np.ndarray      # (T, J) command applied at the frame boundary
    defect_state: np.ndarray      # (T, Z) wrapper state, empty for a bare env
    rewards: np.ndarray           # (T,) transition rewards, stored for robomimic
    dones: np.ndarray             # (T,) transition done flags, normally false
    task_contact: np.ndarray      # (T,) bool, moving body touching its receptacle
    peak_cfrc_ext: float
    mode_codes: np.ndarray        # (T,) uint64, per-step contact/grasp signature
    mode_hash: int                # uint64 fold of mode_codes (9.6.4)
    success_trace: np.ndarray     # (T,) task predicate after each step
    success: bool
    t_success: int                # start of terminal success run, else -1
    controller_config_sha256: str
    clipped: np.ndarray           # (T,) bool, any channel saturated

    @property
    def horizon(self) -> int:
        return len(self.states)

    def rms(self, mask: np.ndarray | None = None) -> float:
        """Position-and-orientation tracking RMS, the 9.4.2 e_RMS.

        Translation is in metres and rotation in radians, so the two are summed
        on the same footing as in the ILC update, whose Gamma is the only place
        the two units are reconciled.
        """
        e = self.error if mask is None else self.error[mask]
        if len(e) == 0:
            raise ValueError("empty mask: e_RMS over no frames is not a number")
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))

    def position_rms(self, mask: np.ndarray | None = None) -> float:
        e = self.error if mask is None else self.error[mask]
        return float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1))))

    def orientation_rms(self, mask: np.ndarray | None = None) -> float:
        e = self.error if mask is None else self.error[mask]
        return float(np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1))))


class OSCTracker:
    """Turns X_tau into the delta actions robosuite's OSC_POSE consumes.

    The map is the exact inverse of what `OperationalSpaceController.set_goal`
    does in `input_type="delta"` mode with `goal_update_mode="achieved"`: the
    controller scales the action by `output_max` and adds it to the currently
    achieved pose expressed in the base frame, so commanding an absolute
    reference pose means dividing the base-frame pose difference by `output_max`.
    Whenever the required delta exceeds one action unit the command saturates,
    which is a real property of the action space and is recorded per frame rather
    than hidden.
    """

    def __init__(self, *, grip_site_candidates: Sequence[str] = (
        "gripper0_right_grip_site", "gripper0_grip_site", "robot0_grip_site",
    )) -> None:
        self._grip_site_candidates = tuple(grip_site_candidates)

    @property
    def gains(self) -> OSCGains:
        """The frozen gains. There is deliberately no setter (9.4.6)."""
        return FROZEN_GAINS

    # -- the action map --------------------------------------------------

    def delta_action(
        self,
        ref_pos: np.ndarray,
        ref_rot: np.ndarray,
        eef_pos: np.ndarray,
        eef_rot: np.ndarray,
        base_pos: np.ndarray,
        base_rot: np.ndarray,
    ) -> np.ndarray:
        """Batched: (N,3),(N,3,3),(N,3),(N,3,3),(3,),(3,3) -> (N,6), unclipped.

        Vectorised over frames on purpose. `query` calls this on the order of
        45 M frames and a per-frame Python loop here would dominate the whole
        ground-truth pass.
        """
        ref_pos = np.atleast_2d(np.asarray(ref_pos, dtype=np.float64))
        eef_pos = np.atleast_2d(np.asarray(eef_pos, dtype=np.float64))
        ref_rot = np.asarray(ref_rot, dtype=np.float64).reshape(-1, 3, 3)
        eef_rot = np.asarray(eef_rot, dtype=np.float64).reshape(-1, 3, 3)
        base_rot = np.asarray(base_rot, dtype=np.float64).reshape(3, 3)
        base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)

        d_world = ref_pos - eef_pos
        d_base = d_world @ base_rot            # base_rot.T @ d, batched
        a_pos = d_base / self.gains.pos_scale

        # The controller composes goal_ori = R_err @ (base_rot.T @ eef_rot) and
        # we want that to equal base_rot.T @ ref_rot, so R_err is the relative
        # rotation conjugated into the base frame.
        rel_world = ref_rot @ np.swapaxes(eef_rot, 1, 2)
        rel_base = base_rot.T[None] @ rel_world @ base_rot[None]
        a_ori = _rotvec(rel_base) / self.gains.ori_scale

        return np.concatenate([a_pos, a_ori], axis=1)

    def compose_action(
        self,
        delta: np.ndarray,
        gripper: np.ndarray,
        feedforward: np.ndarray | None = None,
    ) -> np.ndarray:
        """(N,6) + (N,) [+ (N,6) feedforward in action units] -> (N,7), unclipped.

        9.4.2 puts the ILC feedforward on the *input* of OSC and measures it in
        action units, so the sum happens here and the clip is left to the caller:
        9.6.7 needs the clip applied per side before the difference is taken.
        """
        delta = np.atleast_2d(np.asarray(delta, dtype=np.float64))
        out = np.empty((len(delta), ACTION_DIM), dtype=np.float64)
        out[:, :6] = delta
        if feedforward is not None:
            out[:, :6] += np.asarray(feedforward, dtype=np.float64).reshape(-1, 6)
        out[:, 6] = np.asarray(gripper, dtype=np.float64).reshape(-1)
        return out

    @staticmethod
    def clip_action(action: np.ndarray) -> np.ndarray:
        """Clip to the action range. 9.6.7 item 2: per side, before differencing."""
        return np.clip(action, ACTION_LOW, ACTION_HIGH)

    # -- environment plumbing --------------------------------------------

    def grip_site_id(self, env: Any) -> int:
        for name in self._grip_site_candidates:
            try:
                return int(env.sim.model.site_name2id(name))
            except (ValueError, KeyError):
                continue
        raise ValueError(
            f"none of {self._grip_site_candidates} is a site on this model; the "
            "tracker has to read the same site the arm controller tracks"
        )

    @staticmethod
    def arm_controller(env: Any) -> Any:
        controllers = env.robots[0].composite_controller.part_controllers
        for part in ("right", "left"):
            if part in controllers:
                return controllers[part]
        raise ValueError(f"no arm controller among {sorted(controllers)}")

    def base_frame(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        osc = self.arm_controller(env)
        if osc.origin_pos is None or osc.origin_ori is None:
            raise ValueError(
                "the arm controller has no origin yet; call env.reset() before "
                "asking for the base frame"
            )
        return (
            np.asarray(osc.origin_pos, dtype=np.float64).reshape(3).copy(),
            np.asarray(osc.origin_ori, dtype=np.float64).reshape(3, 3).copy(),
        )

    def tool_pose(self, env: Any, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
        rot = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        return pos, rot

    # -- rollout ----------------------------------------------------------

    def track(
        self,
        env: Any,
        reference: ReferenceTrajectory,
        *,
        family: str,
        seed_env: int,
        feedforward: np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> TrackingResult:
        """Run one closed-loop rollout of X_tau, optionally with an ILC feedforward.

        `noise` is the 9.4.9 demonstration noise epsilon, in action units. It is
        an argument rather than something this module generates because its seed
        is derived from `seed_env` alone and must be bit-identical across a twin
        pair; owning the RNG here would make that a property of the tracker
        instead of a property of the pairing.
        """
        T = reference.horizon
        if T != HORIZON:
            raise ValueError(f"reference has {T} steps, 9.6.1 fixes the horizon at {HORIZON}")
        if feedforward is not None and np.shape(feedforward) != (T, 6):
            raise ValueError(f"feedforward must be ({T}, 6), got {np.shape(feedforward)}")
        if noise is not None and np.shape(noise) != (T, ACTION_DIM):
            raise ValueError(f"noise must be ({T}, {ACTION_DIM}), got {np.shape(noise)}")

        site_id = self.grip_site_id(env)
        base_pos, base_rot = self.base_frame(env)
        nstate = int(env.sim.model.nq) + int(env.sim.model.nv)

        states = np.empty((T, nstate), dtype=np.float64)
        n_actuation = gripper_command_state(env).size
        actuation = np.empty((T, n_actuation), dtype=np.float64)
        actions = np.empty((T, ACTION_DIM), dtype=np.float64)
        actions_raw = np.empty((T, ACTION_DIM), dtype=np.float64)
        pos_cmd = np.empty((T, 3), dtype=np.float64)
        rot_cmd = np.empty((T, 3, 3), dtype=np.float64)
        pos_ach = np.empty((T, 3), dtype=np.float64)
        rot_ach = np.empty((T, 3, 3), dtype=np.float64)
        wrist = np.empty((T, 6), dtype=np.float64)
        actuator_ids = _joint_actuator_ids(env)
        torque_width = len(actuator_ids)
        joint_torque = np.empty((T, torque_width), dtype=np.float64)
        gravity_width = (
            int(np.asarray(env.gravity_torque()).size)
            if callable(getattr(env, "gravity_torque", None))
            else 0
        )
        defect_width = (
            int(np.asarray(env.defect_state()).size)
            if callable(getattr(env, "defect_state", None))
            else 0
        )
        tau_g = np.empty((T, gravity_width), dtype=np.float64)
        defect_state = np.empty((T, defect_width), dtype=np.float64)
        rewards = np.empty(T, dtype=np.float64)
        dones = np.empty(T, dtype=bool)
        contact = np.zeros(T, dtype=bool)
        part, seat = mating_geom_ids(env, task_spec(reference.task_id))
        mode_codes = np.empty(T, dtype=np.uint64)
        peak_cfrc = 0.0
        success_trace = np.empty(T, dtype=bool)

        for t in range(T):
            # ``lite_physics`` can leave derived site fields one integration
            # step behind the qpos/qvel that are now at the control boundary.
            # Refresh kinematics before taking the state snapshot so the pose
            # used to form the command is a deterministic function of the
            # recorded state and can be reproduced by the off-branch query.
            env.sim.forward()
            states[t] = physics_state(env.sim)
            if gravity_width:
                tau_g[t] = env.gravity_torque()
            if defect_width:
                defect_state[t] = env.defect_state()
            row = gripper_command_state(env)
            if row.size != n_actuation:
                raise ValueError(
                    f"the gripper accumulator changed width from {n_actuation} to "
                    f"{row.size} at step {t}. robosuite resets it to "
                    "zeros(gripper.dof) and widens it to one entry per gripper "
                    "actuator on the first control step, so `track` has to start "
                    "from a settled reset (`reset_env`) rather than from a bare "
                    "`env.reset()`"
                )
            actuation[t] = row
            p, r = self.tool_pose(env, site_id)
            pos_cmd[t], rot_cmd[t] = p, r
            delta = self.delta_action(
                reference.positions[t], reference.rotations[t], p, r, base_pos, base_rot
            )
            raw = self.compose_action(
                delta,
                reference.gripper[t: t + 1],
                None if feedforward is None else feedforward[t: t + 1],
            )[0]
            if noise is not None:
                raw = raw + noise[t]
            actions_raw[t] = raw
            actions[t] = self.clip_action(raw)
            wrist[t] = wrist_ft(env)

            _, reward, done, _ = env.step(env_action(env, actions[t]))
            joint_torque[t] = _commanded_joint_torque(env, actuator_ids)
            rewards[t] = reward
            dones[t] = done

            p2, r2 = self.tool_pose(env, site_id)
            pos_ach[t], rot_ach[t] = p2, r2
            if part and seat:
                contact[t] = _touching(env, part, seat)
            mode_codes[t] = mode_code(env)
            peak_cfrc = max(peak_cfrc, float(np.max(np.linalg.norm(env.sim.data.cfrc_ext, axis=1))))
            success_trace[t] = bool(env._check_success())

        t_success = terminal_hold_start(success_trace)

        error = np.empty((T, 6), dtype=np.float64)
        error[:, :3] = reference.positions - pos_ach
        error[:, 3:] = _rotvec(reference.rotations @ np.swapaxes(rot_ach, 1, 2))

        return TrackingResult(
            task_id=reference.task_id,
            family=family,
            seed_env=seed_env,
            states=states,
            actuation=actuation,
            actions=actions,
            actions_raw=actions_raw,
            eef_pos_cmd=pos_cmd,
            eef_rot_cmd=rot_cmd,
            eef_pos_ach=pos_ach,
            eef_rot_ach=rot_ach,
            error=error,
            wrist_ft=wrist,
            tau_g=tau_g,
            joint_torque=joint_torque,
            defect_state=defect_state,
            rewards=rewards,
            dones=dones,
            task_contact=contact,
            peak_cfrc_ext=peak_cfrc,
            mode_codes=mode_codes,
            mode_hash=fold_mode_codes(mode_codes),
            success_trace=success_trace,
            success=t_success >= 0,
            t_success=t_success,
            controller_config_sha256=self.gains.sha256(),
            clipped=np.any(np.abs(actions_raw) > ACTION_HIGH, axis=1),
        )


def wrist_ft(env: Any) -> np.ndarray:
    """The 6-dim wrist force/torque channel of the 9.6.1 proprio vector."""
    robot = env.robots[0]
    force = robot.ee_force
    torque = robot.ee_torque
    if isinstance(force, dict):
        arm = "right" if "right" in force else next(iter(force))
        force, torque = force[arm], torque[arm]
    return np.concatenate([np.asarray(force, dtype=np.float64),
                           np.asarray(torque, dtype=np.float64)])


def _body_geom_ids(env: Any, body: str) -> frozenset[int]:
    model = env.sim.model
    bid = model.body_name2id(body)
    adr = int(model.body_geomadr[bid])
    return frozenset(range(adr, adr + int(model.body_geomnum[bid])))


def mating_geom_ids(env: Any, spec: TaskSpec) -> tuple[frozenset[int], frozenset[int]]:
    """The two geom sets whose contact means the part has engaged its receptacle.

    The pair is the one 9.5.4 already names for `compute_task_tol`: the moving
    body and whichever of the two measured bodies it is not. A task that declares
    no such pair -- Door, Wipe, Lift -- has no mating event, and both sets come
    back empty rather than being stood in for by some other contact.
    """
    if spec.moving_body is None:
        return frozenset(), frozenset()
    other = spec.inserted_body if spec.moving_body == spec.hole_body else spec.hole_body
    if other is None:
        return frozenset(), frozenset()
    return _body_geom_ids(env, spec.moving_body), _body_geom_ids(env, other)


def _touching(env: Any, a: frozenset[int], b: frozenset[int]) -> bool:
    """Is any geom of `a` in contact with any geom of `b`?

    This is the contact test the ILC degradation rule keys on (9.4.2). It is
    deliberately geometric rather than the r4 regime label: r4 is defined by the
    wrist force clearing `f_th`, a per-cell threshold that 9.7.1 calibrates from
    rollouts, and keying the teacher's own iteration on it would put a
    measurement in front of the rollouts that produce it. Touching or not
    touching needs no threshold.
    """
    data = env.sim.data
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in a and g2 in b) or (g2 in a and g1 in b):
            return True
    return False


def mode_code(env: Any) -> np.uint64:
    """A per-step contact signature: the set of bodies currently touching.

    9.6.4 defines the mode sequence hash as contact-pair sequence plus grasp id
    plus the side taken around an obstacle. The first two are computed here; the
    third is only defined on the G2 baffle variant and is contributed by the
    geometry layer, so `fold_mode_codes` accepts an extra per-step code from the
    caller rather than this function inventing one.

    Pairs are taken at body level, not geom level. Which bodies are in contact is
    the discrete mode; which of a body's collision primitives happens to carry
    the contact is a meshing detail that flickers several times per control step
    while a jaw settles onto a part, and at geom level that chatter alone makes
    two runs of the same trajectory disagree -- the mode-stability condition
    (C3) then never holds and every demonstration is thrown away for a reason
    that has nothing to do with contact mode.
    """
    data = env.sim.data
    n = int(data.ncon)
    if n == 0:
        return np.uint64(0xcbf29ce484222325)
    bodies = env.sim.model.geom_bodyid
    pairs = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        c = data.contact[i]
        b1, b2 = int(bodies[c.geom1]), int(bodies[c.geom2])
        pairs[i] = (b1, b2) if b1 <= b2 else (b2, b1)
    pairs = np.unique(pairs, axis=0)
    h = hashlib.blake2b(pairs.tobytes(), digest_size=8)
    return np.frombuffer(h.digest(), dtype=np.uint64)[0]


def mode_sequence(codes: np.ndarray) -> np.ndarray:
    """Collapse a per-step code stream to the sequence of distinct modes it visits.

    4.3 and 9.6.4 both call the quantity a contact-pair *sequence*, and (A3) uses
    it to ask whether a defective trajectory and its twin pass through the same
    discrete modes. That question is about which contacts occur in which order,
    not about how many control steps each one lasted: a defect that slows the
    approach by one step changes every per-step sample while changing no mode at
    all, and hashing the raw stream would exclude the entire study on timing
    jitter. Run-length collapsing is what makes the hash answer the question that
    was asked.
    """
    codes = np.asarray(codes, dtype=np.uint64)
    if len(codes) == 0:
        return codes
    keep = np.empty(len(codes), dtype=bool)
    keep[0] = True
    np.not_equal(codes[1:], codes[:-1], out=keep[1:])
    return codes[keep]


def fold_mode_codes(codes: np.ndarray, extra: np.ndarray | None = None) -> int:
    """Fold a per-step mode code stream into the mode sequence hash of 9.6.4.

    `extra` carries the obstacle-side component for the G2 geometry variant,
    which only the geometry layer can produce. Folding it in here keeps one
    definition of the hash instead of two that could drift apart. It is
    collapsed on its own, because the side taken around an obstacle changes on
    its own schedule and pairing the two streams step by step would reintroduce
    the timing sensitivity that collapsing removes.
    """
    payload = mode_sequence(codes).tobytes()
    if extra is not None:
        payload += mode_sequence(np.asarray(extra)).tobytes()
    h = hashlib.blake2b(payload, digest_size=8)
    return int(np.frombuffer(h.digest(), dtype=np.uint64)[0])


# ---------------------------------------------------------------------------
# 9.4.6: the one-off gain selection, and the determinism check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GainTuningReport:
    """Result of the 9.4.6 grid. Reports; never rewrites `FROZEN_GAINS`."""

    grid: tuple[float, ...]
    rms_by_kp: dict[float, float]
    tasks: tuple[str, ...]
    families: tuple[str, ...]
    seeds: tuple[int, ...]
    frozen_kp: float = FROZEN_KP

    @property
    def argmin_kp(self) -> float:
        return min(self.rms_by_kp, key=lambda k: self.rms_by_kp[k])

    @property
    def agrees_with_frozen(self) -> bool:
        return self.argmin_kp == self.frozen_kp

    def render(self) -> str:
        lines = [
            "9.4.6 tracker gain selection (clean twins only)",
            f"  tasks    : {', '.join(self.tasks)}",
            f"  families : {', '.join(self.families)}",
            f"  seeds    : {', '.join(str(s) for s in self.seeds)}",
            "  kp      e_RMS (m+rad, pooled)",
        ]
        for kp in self.grid:
            mark = "  <- frozen" if kp == self.frozen_kp else ""
            lines.append(f"  {kp:6.0f}  {self.rms_by_kp[kp]:.6f}{mark}")
        lines.append(
            f"  argmin = {self.argmin_kp:.0f}, frozen = {self.frozen_kp:.0f}: "
            + ("agree" if self.agrees_with_frozen else
               "DISAGREE -- this is a write-back to 9.1.1/9.4.2, not a runtime override")
        )
        return "\n".join(lines)


def tune_gains(
    *,
    tasks: Iterable[str],
    families: Iterable[str],
    seeds: Iterable[int],
    grid: Sequence[float] = KP_GRID,
) -> GainTuningReport:
    """The 9.4.6 grid: pick kp once on the clean twins, by pooled tracking RMS.

    Run on clean twins only, by construction: no defect is injected anywhere in
    this function. The result is a report. Changing what rollouts use means
    editing `FROZEN_KP` and re-freezing, which is a visible diff -- 9.4.6's whole
    point is that a gain change must not be expressible as a runtime argument.
    """
    tasks = tuple(tasks)
    families = tuple(families)
    seeds = tuple(int(s) for s in seeds)
    tracker = OSCTracker()
    rms: dict[float, float] = {}
    for kp in grid:
        gains = OSCGains(kp=kp)
        total_sq, total_n = 0.0, 0
        for task_id in tasks:
            spec = load_spec(task_id)
            for fam in families:
                for seed in seeds:
                    env = make_env(task_id, family=fam, seed_env=seed, gains=gains)
                    try:
                        reference = resolve(spec, env.sim)
                        # The tracker is the same object across the grid; only the
                        # controller inside the env changes, which is the point.
                        res = _track_with_gains(tracker, env, reference, gains, fam, seed)
                        total_sq += float(np.sum(res.error * res.error))
                        total_n += res.horizon
                    finally:
                        env.close()
        rms[float(kp)] = float(np.sqrt(total_sq / total_n))
    return GainTuningReport(
        grid=tuple(float(k) for k in grid),
        rms_by_kp=rms,
        tasks=tasks,
        families=families,
        seeds=seeds,
    )


def _track_with_gains(
    tracker: OSCTracker,
    env: Any,
    reference: ReferenceTrajectory,
    gains: OSCGains,
    family: str,
    seed_env: int,
) -> TrackingResult:
    """Run the tracker against an env whose controller uses `gains`.

    Only the tuner needs this. The scale conversion in `delta_action` reads
    `self.gains`, and every entry of `KP_GRID` shares the `output_max` of
    `FROZEN_GAINS`, so the action map is identical across the grid and the grid
    varies exactly one thing: the impedance the environment applies.
    """
    if gains.output_max != FROZEN_GAINS.output_max:
        raise ValueError(
            "the 9.4.6 grid varies kp only. Changing output_max changes the action "
            "unit, and 9.6.1's '1 unit = 50 mm / 0.5 rad' ruler is what every "
            "magnitude in the proposal is quoted against"
        )
    return tracker.track(env, reference, family=family, seed_env=seed_env)


@dataclass
class DeterminismCheck:
    task_id: str
    family: str
    seed_env: int
    states_bit_exact: bool
    actions_bit_exact: bool
    max_state_diff: float
    max_action_diff: float

    @property
    def ok(self) -> bool:
        return self.states_bit_exact and self.actions_bit_exact

    def render(self) -> str:
        verdict = "bit-exact" if self.ok else "NOT bit-exact"
        return (
            f"9.1.6 determinism, {self.family}/{self.task_id}/seed={self.seed_env}: "
            f"{verdict} (max |dstate| = {self.max_state_diff:g}, "
            f"max |daction| = {self.max_action_diff:g})"
        )


def assert_deterministic(
    task_id: str, *, family: str, seed_env: int, raise_on_failure: bool = True
) -> DeterminismCheck:
    """Run the same cell twice from scratch and compare states and actions.

    9.1.6 makes this a requirement rather than an expectation, and 11.2 puts the
    highest determinism bar on `teacher/` specifically because every ground truth
    comes from it.
    """
    tracker = OSCTracker()
    spec = load_spec(task_id)
    runs = []
    for _ in range(2):
        env = make_env(task_id, family=family, seed_env=seed_env)
        try:
            runs.append(tracker.track(env, resolve(spec, env.sim), family=family,
                                      seed_env=seed_env))
        finally:
            env.close()
    ds = float(np.max(np.abs(runs[0].states - runs[1].states)))
    da = float(np.max(np.abs(runs[0].actions - runs[1].actions)))
    check = DeterminismCheck(
        task_id=task_id,
        family=family,
        seed_env=seed_env,
        states_bit_exact=bool(np.array_equal(runs[0].states, runs[1].states)),
        actions_bit_exact=bool(np.array_equal(runs[0].actions, runs[1].actions)),
        max_state_diff=ds,
        max_action_diff=da,
    )
    if raise_on_failure and not check.ok:
        raise AssertionError(
            check.render()
            + "\n9.1.6 requires bit-exact reproduction from (embodiment, task, "
            "seed_env). A near-miss is still a failure: it means some state "
            "outside seed_env is leaking into the rollout"
        )
    return check
