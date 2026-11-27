"""W1 calibration: the 待核实 constants of chapter 9.

Every quantity this script produces is one the proposal refuses to let anything
downstream assume. 9.8.5 orders them by what they block rather than by section
number, and the `--what` subcommands below follow that order:

    tau_step, tau_frame, train_throughput   -> the whole budget (9.8), and the
                                               decision whether to port to MJX
    proprio_dim                             -> every hdf5 shape and the policy
                                               input width
    armature                                -> D4's shadow-joint inertia and the
                                               FGW inertia feature
    hard_reset                              -> whether model-level injection has
                                               to survive an XML recompile
    camera_pose                             -> image reproducibility
    gzip_ratio                              -> the 758 GB storage total
    defect_observable, stiction             -> the dose axes of D5 and D6
    d7_stability                            -> D7's clean twin, which the explicit
                                               stiffness integrator caps
    clearance                               -> cl_pm, proposition 2(b), T8
    tau_corr                                -> effective sample size, hence every
                                               design-graph quantity
    phi_rank                                -> the E_phi gate

Results are written to `registry/lock.json` with provenance. Nothing else in the
package computes any of them, and nothing reads one except through
`portal_trace.lockfile`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from portal_trace.env.versions import Backend, check_pinned_versions, fingerprint
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile, source_commit
from portal_trace.registry.tasks import (
    CONTROL_FREQ,
    HORIZON,
    SIMULATION_TIMESTEP,
    TASKS,
    TaskSpec,
    task,
)
from portal_trace.types import SeedTriple

__all__ = ["main", "measure_tau_step", "measure_proprio_dim", "measure_armature"]

# The families that can be probed without the registry module: the six-family
# core is assembled there, but tau_step and the model facts are per-robot and
# every robosuite robot name works here.
_DEFAULT_FAMILY = "Panda"


def _make_env(spec: TaskSpec, family: str, *, cameras: tuple[str, ...] = ()):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    if spec.tight_variant_of is not None:
        # Tight-fit tasks are project-defined subclasses and are not available
        # in robosuite's registry until this explicit construction step.  The
        # measurement runner must use the same registration boundary as runtime
        # collection; otherwise its timing/physics constants fail only for the
        # exact tasks that gate the tolerance calibration.
        from portal_trace.registry.tight import ensure_registered

        ensure_registered(spec)
    kwargs = dict(spec.env_kwargs)
    kwargs.pop("horizon", None)
    controller = load_composite_controller_config(controller=None, robot=family)
    return suite.make(
        spec.env_name,
        robots=family,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=bool(cameras),
        use_camera_obs=bool(cameras),
        camera_names=list(cameras) if cameras else None,
        camera_heights=84,
        camera_widths=84,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        ignore_done=True,
        **kwargs,
    )


@dataclass(frozen=True)
class Timing:
    """Both the mean and the median, because they answer different questions.

    The budget of 9.8 multiplies a per-step cost by a step count, so the mean is
    the number that belongs in it. The median is reported next to it because on
    a shared machine the mean picks up scheduler spikes, and a mean far above the
    median is the signal that the measurement environment, not the simulator,
    dominates -- in which case the number should be re-taken on the cluster
    before the budget is trusted.
    """

    seconds_per_call: float
    median_seconds: float
    n_calls: int
    n_warmup: int
    std_seconds: float


def _time_steps(env, n_steps: int, n_warmup: int) -> Timing:
    """Wall clock per env.step with a fixed action, warm-up discarded.

    A fixed action rather than a random one: 9.8.4 wants the physics cost, and a
    random action wanders into contact-heavy states at a rate that depends on the
    RNG, which would make the number depend on the seed rather than on the task.
    """
    action = np.zeros(env.action_dim)
    for _ in range(n_warmup):
        env.step(action)
    samples = np.empty(n_steps)
    for i in range(n_steps):
        t0 = time.perf_counter()
        env.step(action)
        samples[i] = time.perf_counter() - t0
    return Timing(
        seconds_per_call=float(samples.mean()),
        median_seconds=float(np.median(samples)),
        n_calls=n_steps,
        n_warmup=n_warmup,
        std_seconds=float(samples.std(ddof=1)),
    )


def measure_tau_step(task_id: str, family: str, *, n_steps: int, n_warmup: int) -> Timing:
    env = _make_env(task(task_id), family)
    try:
        env.reset()
        return _time_steps(env, n_steps, n_warmup)
    finally:
        env.close()


def measure_tau_frame(task_id: str, family: str, *, n_steps: int, n_warmup: int) -> Timing:
    """Per-step cost with the two 84x84 cameras on, minus the physics-only cost.

    9.6.1 fixes the two camera names, and 9.1.3's rendering line counts 400 x 2
    frames per episode, so what the budget needs is the marginal cost of a
    rendered frame rather than the total.
    """
    cameras = ("agentview", "robot0_eye_in_hand")
    env = _make_env(task(task_id), family, cameras=cameras)
    try:
        env.reset()
        with_render = _time_steps(env, n_steps, n_warmup)
    finally:
        env.close()
    physics_only = measure_tau_step(task_id, family, n_steps=n_steps, n_warmup=n_warmup)
    marginal = (with_render.seconds_per_call - physics_only.seconds_per_call) / len(cameras)
    if marginal <= 0:
        raise RuntimeError(
            f"rendering measured as free ({with_render.seconds_per_call:.6g} s with "
            f"cameras vs {physics_only.seconds_per_call:.6g} s without); the timing is "
            "dominated by noise, raise --n-steps rather than recording this"
        )
    return Timing(
        seconds_per_call=marginal,
        median_seconds=(with_render.median_seconds - physics_only.median_seconds) / len(cameras),
        n_calls=with_render.n_calls,
        n_warmup=with_render.n_warmup,
        std_seconds=float(
            np.hypot(with_render.std_seconds, physics_only.std_seconds) / len(cameras)
        ),
    )


def measure_proprio_dim(family: str) -> dict[str, Any]:
    """The native width of `robot0_proprio-state`, and the resulting layout.

    9.6.1 assumes 32 and says in the same breath that if it is not 32 then the
    52 and the hdf5 shapes change with it. It is not 32.
    """
    env = _make_env(task("NutAssemblySquare"), family)
    try:
        obs = env.reset()
        native = int(np.asarray(obs["robot0_proprio-state"]).shape[0])
        components = {
            k: int(np.asarray(v).shape[0])
            for k, v in sorted(obs.items())
            if k.startswith("robot0_") and k != "robot0_proprio-state"
        }
        n_arm_joints = int(np.asarray(obs["robot0_joint_pos"]).shape[0])
    finally:
        env.close()
    # 9.6.1's own arithmetic, recomputed with the measured native width: the
    # gravity-torque and wrist force/torque channels are what the 9.7 regime
    # decision rules read, so they have to be in the policy input for (A5)(ii)
    # to hold, and the DoF mask rides along.
    n_tau_g = n_arm_joints
    n_wrist = 6
    n_mask = n_arm_joints
    return {
        "native": native,
        "components": components,
        "n_arm_joints": n_arm_joints,
        "physical_channels": native + n_tau_g + n_wrist,
        "stored": native + n_tau_g + n_wrist + n_mask,
        "breakdown": f"{native} native + {n_tau_g} tau_g + {n_wrist} wrist F/T + {n_mask} DoF mask",
    }


def measure_armature(family: str) -> dict[str, float]:
    env = _make_env(task("NutAssemblySquare"), family)
    try:
        env.reset()
        model = env.sim.model._model
        joints = env.robots[0].robot_joints
        out = {}
        for name in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = int(model.jnt_dofadr[jid])
            out[name] = {
                "armature": float(model.dof_armature[dof]),
                "damping": float(model.dof_damping[dof]),
                "frictionloss": float(model.dof_frictionloss[dof]),
            }
    finally:
        env.close()
    return out


def measure_actuator_limits(family: str) -> dict[str, Any]:
    """ctrlrange and forcerange per actuator: D5's dead zone is a fraction of tau_max.

    9.2.5 states the dead zone as w / tau_max, so tau_max has to be read off the
    model rather than assumed. MuJoCo ignores `forcerange` unless `forcelimited`
    is set, so both are reported and the caller has to look at the flag.
    """
    env = _make_env(task("NutAssemblySquare"), family)
    try:
        env.reset()
        model = env.sim.model._model
        out = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
            out.append({
                "name": name,
                "ctrlrange": [float(x) for x in model.actuator_ctrlrange[i]],
                "ctrllimited": bool(model.actuator_ctrllimited[i]),
                "forcerange": [float(x) for x in model.actuator_forcerange[i]],
                "forcelimited": bool(model.actuator_forcelimited[i]),
                "gainprm0": float(model.actuator_gainprm[i, 0]),
                "gaintype": int(model.actuator_gaintype[i]),
                "biastype": int(model.actuator_biastype[i]),
            })
    finally:
        env.close()
    return {"actuators": out}


def measure_hard_reset_default() -> dict[str, Any]:
    """Whether robosuite rebuilds `sim` from XML on reset.

    It does, by default. That makes `_apply_model_level` mandatory after every
    reset rather than a construction-time step, which is the whole reason 9.1.4
    marks it 待核实.
    """
    import inspect

    from robosuite.environments.base import MujocoEnv

    sig = inspect.signature(MujocoEnv.__init__)
    return {
        "hard_reset": bool(sig.parameters["hard_reset"].default),
        "lite_physics": bool(sig.parameters["lite_physics"].default),
    }


def measure_camera_pose(task_id: str, family: str) -> dict[str, Any]:
    env = _make_env(task(task_id), family)
    try:
        env.reset()
        model = env.sim.model._model
        data = env.sim.data._data
        out = {}
        for name in ("agentview", "robot0_eye_in_hand"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                continue
            out[name] = {
                "pos": [float(x) for x in data.cam_xpos[cid]],
                "mat": [float(x) for x in np.asarray(data.cam_xmat[cid]).reshape(-1)],
                "fovy": float(model.cam_fovy[cid]),
            }
        if not out:
            raise RuntimeError(f"{task_id}: neither camera exists in the compiled model")
    finally:
        env.close()
    return out


def measure_simulation_facts(task_id: str, family: str) -> dict[str, Any]:
    env = _make_env(task(task_id), family)
    try:
        env.reset()
        model = env.sim.model._model
        substeps = env.control_timestep / model.opt.timestep
        return {
            "timestep": float(model.opt.timestep),
            "control_freq": float(env.control_freq),
            "substeps_per_control_step": float(substeps),
            "injection_hook_hz": float(1.0 / model.opt.timestep),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "impratio": float(model.opt.impratio),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "solver": int(model.opt.solver),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        }
    finally:
        env.close()


# The rigid twin's mode has to sit far enough below the integrator's limit that a
# 400-step episode with contact does not creep over it, and far enough above the
# defective mode that D7 is a defect at all. 0.75 of the last stable grid point is
# the margin taken for the former; the latter comes out of the search as the
# integer ratio N and is reported so it can be read off.
D7_STABILITY_MARGIN = 0.75


def _diverged(env: Any) -> bool:
    """MuJoCo's own verdict, not a magnitude threshold picked here."""
    data = env.sim.data
    for warn in (
        mujoco.mjtWarning.mjWARN_BADQACC,
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
    ):
        if data.warning[warn].number > 0:
            return True
    return not (np.all(np.isfinite(data.qacc)) and np.all(np.isfinite(data.qpos)))


def measure_d7_stability(
    task_id: str, family: str, *, n_steps: int, n_grid: int
) -> dict[str, Any]:
    """Where the explicit stiffness integrator gives out on D7's flexible joint.

    D7's fragment sets stiffness per unit of compiled inertia, so the mode sits at
    omega = omega_ref / sqrt(ratio) whatever the arm carries -- the sweep is over
    omega * dt, which is the quantity the Euler step actually constrains, and the
    compliance ratio is read back off it. Only the defective side is built:
    `Delta.zero()` on D7 would ask for the very number being measured.
    """
    from portal_trace.defects.delta import Delta, all_joint_scale
    from portal_trace.defects.hooks.shadow_joint import LINK_FLEX_REFERENCE_OMEGA
    from portal_trace.defects.inject import build_single

    spec = task(task_id)
    dt = SIMULATION_TIMESTEP
    unit = LINK_FLEX_REFERENCE_OMEGA * dt  # omega * dt at ratio = 1, the defective dose
    seeds = SeedTriple(env=0, train=0, eval=0)
    probe = _make_env(spec, family)
    try:
        n_joints = len(probe.robots[0].robot_joints)
    finally:
        probe.close()

    # Walk stiffness upward -- compliance downward -- until MuJoCo complains.
    grid = [unit * n for n in range(1, n_grid + 1)]
    trials: list[dict[str, Any]] = []
    last_stable: float | None = None
    first_divergent: float | None = None
    for omega_dt in grid:
        ratio = (unit / omega_dt) ** 2
        delta = Delta(
            defect_code="D7",
            level=0,
            params={"link_compliance": ratio},
            # Every arm joint, not the single-joint default: the bound wanted here
            # is the one that holds when all seven flexible modes run at once.
            joint_scale=all_joint_scale(n_joints),
        )
        wrapper = build_single(spec, family, delta, seeds)
        try:
            wrapper.reset()
            action = np.zeros(wrapper.env.action_dim)
            steps_survived = 0
            for _ in range(n_steps):
                wrapper.step(action)
                steps_survived += 1
                if _diverged(wrapper.env):
                    break
            bad = _diverged(wrapper.env)
        finally:
            wrapper.env.close()
        trials.append(
            {
                "omega_dt": omega_dt,
                "link_compliance": ratio,
                "n_over_defective": round(omega_dt / unit),
                "steps_survived": steps_survived,
                "diverged": bad,
            }
        )
        if bad:
            first_divergent = omega_dt
            break
        last_stable = omega_dt

    if last_stable is None:
        raise RuntimeError(
            f"D7 diverged at the first grid point, omega dt = {grid[0]:.4f}, which is "
            "the defective dose itself. The defect cannot be built at this timestep; "
            "9.2.7's D7 row has to be revisited before any D7 cell can run."
        )
    if first_divergent is None:
        raise RuntimeError(
            f"the sweep reached omega dt = {last_stable:.4f} without diverging. The "
            "boundary has to be bracketed, not extrapolated: rerun with a larger "
            f"--n-grid than {n_grid}."
        )

    n_clean = int(D7_STABILITY_MARGIN * last_stable / unit)
    if n_clean < 2:
        raise RuntimeError(
            f"the stable band admits only N = {n_clean}; the rigid twin would sit at "
            "or below the defective mode, which inverts the defect"
        )
    return {
        "link_compliance": 1.0 / float(n_clean) ** 2,
        "n_over_defective": n_clean,
        "omega_dt_clean": n_clean * unit,
        "omega_dt_last_stable": last_stable,
        "omega_dt_first_divergent": first_divergent,
        "omega_dt_defective": unit,
        "omega_ref": LINK_FLEX_REFERENCE_OMEGA,
        "timestep": dt,
        "margin": D7_STABILITY_MARGIN,
        "n_steps": n_steps,
        "trials": trials,
    }


# 9.2.6 trap ii, and the criterion of the contingency clause: motion is declared
# at 1e-3 rad/s, and D6 leaves the 36 twin pairs if the offset between the two
# calibers exceeds half of the smallest dose, 0.05 N m.
STICTION_VELOCITY_BAND = 1.0e-3
STICTION_REJECT_ABOVE = 0.5 * 0.05


def measure_stiction(
    family: str,
    *,
    n_grid: int,
    hold_time_s: float,
    joints: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """W1-8: the known static-friction offset of D6's `qfrc_applied` caliber (9.2.6).

    9.2.6's contingency moves D6's symmetric component out of `dof_frictionloss`
    and into `qfrc_applied`, so that D6 writes nothing at `mjModel` level and the
    twin pair stays bit-identical even under a baseline feature set that reads
    friction. The clause states its own cost and refuses to leave it unmeasured:
    the constraint implementation resists *before motion begins*, a sign-dependent
    force cannot, and the difference is a stiction offset that has to be reported.

    The protocol is 9.2.6's, verbatim. On one configuration, a ladder of
    increasing quasi-static torque steps; report the torque at which `|qdot_j|`
    first exceeds 1e-3 rad/s under each caliber; the difference is the offset. If
    it exceeds half the smallest dose, 0.025 N m, D6 leaves the 36 twin pairs and
    a seventh defect takes its place.

    Three things are held away from the joint under test so that what is measured
    is friction and not something else. Gravity is switched off, or the threshold
    would be the torque needed to overcome the arm's own weight, which has
    nothing to do with either caliber. Actuation is disabled through
    `mjDSBL_ACTUATION` rather than by zeroing `ctrl`, because a zero setpoint on a
    position-like actuator is a command to hold, not an absence of force. And the
    configuration is the arm's `init_qpos` rather than the compiled `qpos0`: the
    latter is all zeros, which is *outside* Panda's joint 4 range of
    [-3.0718, -0.0698] and within the solver's margin of several others, so a
    probe started there is pushed by twelve active limit constraints and breaks
    away at the first rung of the ladder whatever the friction bound is. The
    configuration is checked for exactly that rather than trusted.

    The reported threshold is a property of the ladder's resolution and of the
    hold time, both of which are recorded next to it. The hold is quasi-static in
    the sense that matters here: it is long enough for the constraint caliber's
    verdict to be the friction bound rather than the transient, which is checked
    against the dof's own time constant rather than asserted.
    """
    import mujoco as mj

    from portal_trace.defects.inject import D6_SYMMETRIC_FRACTION, ZERO_VELOCITY_BAND
    from portal_trace.registry.families import descriptor

    if ZERO_VELOCITY_BAND != STICTION_VELOCITY_BAND:
        raise ValueError(
            f"the injector's dead band is {ZERO_VELOCITY_BAND} and this measurement "
            f"uses {STICTION_VELOCITY_BAND}. 9.2.6 fixes one number for both: the "
            "band that suppresses sign chatter is the same band that defines when "
            "motion has begun, and measuring the offset at a different one reports "
            "an offset the injector does not have"
        )

    desc = descriptor(family)
    robot = _robot_model(family)
    model = robot.get_model(mode="mujoco")
    data = mj.MjData(model)
    q0 = np.asarray(robot.init_qpos, dtype=np.float64)
    if q0.shape != (model.nq,):
        raise ValueError(
            f"{family}'s init_qpos has {q0.shape} entries and the model has "
            f"{model.nq}; the probe configuration cannot be set"
        )
    _assert_unconstrained(model, data, q0, family)

    names = joints if joints is not None else tuple(robot.joints)
    nodes = {n.name: n for n in desc.nodes}
    out: dict[str, Any] = {}

    for name in names:
        node = nodes.get(name)
        if node is None:
            raise KeyError(f"{family} has no joint node {name!r}; it has {sorted(nodes)}")
        lo, hi = node.torque_limit
        rated = max(abs(lo), abs(hi))
        if rated <= 0.0:
            raise ValueError(
                f"{family}/{name} has a zero torque limit, so D6's f_bar = "
                f"{D6_SYMMETRIC_FRACTION} x rated is zero and the joint carries no "
                "symmetric friction to measure"
            )
        f_bar = D6_SYMMETRIC_FRACTION * rated
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        dof = int(model.jnt_dofadr[jid])

        # The ladder brackets f_bar on both sides: the constraint caliber breaks
        # away at f_bar, the force caliber somewhere below it, and a ladder that
        # stopped at f_bar could not tell "at the bound" from "above it".
        grid = np.linspace(0.0, 2.0 * f_bar, n_grid + 1)[1:]
        thresholds: dict[str, float | None] = {}
        traces: dict[str, list[dict[str, float]]] = {}
        for caliber in ("constraint", "force"):
            trace: list[dict[str, float]] = []
            hit: float | None = None
            for torque in grid:
                peak = _stiction_trial(
                    model,
                    data,
                    q0,
                    dof=dof,
                    torque=float(torque),
                    f_bar=f_bar,
                    caliber=caliber,
                    hold_time_s=hold_time_s,
                )
                trace.append({"torque": float(torque), "peak_qvel": peak})
                if hit is None and peak > STICTION_VELOCITY_BAND:
                    hit = float(torque)
            thresholds[caliber] = hit
            traces[caliber] = trace

        if thresholds["constraint"] is None or thresholds["force"] is None:
            never = [k for k, v in thresholds.items() if v is None]
            raise RuntimeError(
                f"{family}/{name}: the {never} caliber never broke away up to "
                f"{grid[-1]:.4f} N m, twice f_bar. The ladder does not bracket the "
                "threshold, so the offset is a bound and not a measurement; rerun "
                "with a longer --stiction-hold or a wider ladder"
            )
        offset = float(thresholds["constraint"] - thresholds["force"])
        step = float(grid[1] - grid[0])
        out[name] = {
            "offset": offset,
            "threshold_constraint": thresholds["constraint"],
            "threshold_force": thresholds["force"],
            "f_bar": f_bar,
            "rated_torque": rated,
            "armature": node.armature,
            "damping": node.damping,
            "native_frictionloss": node.frictionloss,
            "ladder_step": step,
            "hold_time_s": hold_time_s,
            # The dof's own time constant. A hold shorter than a few of these
            # would report the transient rather than the friction bound.
            "settling_time_s": (
                float(node.armature / node.damping) if node.damping > 0.0 else None
            ),
            "rejects_d6": abs(offset) > STICTION_REJECT_ABOVE,
            "trace": traces,
        }
    return out


def _assert_unconstrained(model: Any, data: Any, q0: np.ndarray, family: str) -> None:
    """Refuse a probe configuration that is already pushing against something.

    Only friction constraints may be active at the start. A limit or contact
    constraint at the probe pose supplies its own torque, and the ladder then
    measures how hard the solver is pushing rather than where friction gives way.
    """
    import mujoco as mj

    saved_gravity = np.array(model.opt.gravity)
    saved_disable = int(model.opt.disableflags)
    try:
        model.opt.gravity[:] = 0.0
        model.opt.disableflags = saved_disable | int(mj.mjtDisableBit.mjDSBL_ACTUATION)
        mj.mj_resetData(model, data)
        data.qpos[:] = q0
        mj.mj_forward(model, data)
        offenders: dict[str, int] = {}
        for i in range(int(data.nefc)):
            kind = int(data.efc_type[i])
            if kind == int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF):
                continue
            name = str(mj.mjtConstraint(kind))
            offenders[name] = offenders.get(name, 0) + 1
        if offenders:
            raise ValueError(
                f"{family}'s probe configuration has active non-friction "
                f"constraints {offenders}. 9.2.6 asks for a quasi-static torque "
                "ladder on one configuration; started against a limit or a "
                "contact, every rung breaks away and the threshold is the "
                "solver's, not friction's"
            )
    finally:
        model.opt.gravity[:] = saved_gravity
        model.opt.disableflags = saved_disable
        mj.mj_resetData(model, data)


def _stiction_trial(
    model: Any,
    data: Any,
    q0: np.ndarray,
    *,
    dof: int,
    torque: float,
    f_bar: float,
    caliber: str,
    hold_time_s: float,
) -> float:
    """Peak `|qdot_dof|` under one torque step, on a model restored afterwards."""
    import mujoco as mj

    saved_gravity = np.array(model.opt.gravity)
    saved_disable = int(model.opt.disableflags)
    saved_friction = float(model.dof_frictionloss[dof])
    try:
        model.opt.gravity[:] = 0.0
        model.opt.disableflags = saved_disable | int(mj.mjtDisableBit.mjDSBL_ACTUATION)
        model.dof_frictionloss[dof] = f_bar if caliber == "constraint" else 0.0

        mj.mj_resetData(model, data)
        data.qpos[:] = q0
        mj.mj_forward(model, data)
        n_steps = max(1, int(round(hold_time_s / model.opt.timestep)))
        peak = 0.0
        for _ in range(n_steps):
            # qfrc_applied is not cleared by mj_step (9.1.1), and the sign-
            # dependent term has to be recomputed every substep anyway, so it is
            # rewritten rather than set once.
            applied = torque
            if caliber == "force":
                qdot = float(data.qvel[dof])
                sign = np.sign(qdot) if abs(qdot) > STICTION_VELOCITY_BAND else 0.0
                applied -= f_bar * sign
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof] = applied
            mj.mj_step(model, data)
            peak = max(peak, abs(float(data.qvel[dof])))
        return peak
    finally:
        model.opt.gravity[:] = saved_gravity
        model.opt.disableflags = saved_disable
        model.dof_frictionloss[dof] = saved_friction
        data.qfrc_applied[:] = 0.0
        mj.mj_resetData(model, data)


def _robot_model(family: str):
    from portal_trace.registry.families import family as get_family

    return get_family(family).robot_model()


def measure_clearance(
    task_id: str,
    family: str,
    *,
    variants: tuple[str, ...] | None = None,
    well_depth: float | None = None,
    baffle_height: float | None = None,
) -> dict[str, Any]:
    """W1-4: cl_+ and cl_- at the terminal pose, per geometry variant (9.5.5).

    The pair is what T8's harm asymmetry `A = cl_- - cl_+` is built from, and it
    is also how 9.5.5's d and h are checked: the well is *specified* as d deep,
    but the lead-in eats into the top of its wall, so the clearance the stepped
    collision query finds on the built geometry is the number that goes in the
    lock, not the nominal one.

    x* is measured once, on G0, and reused unchanged for every variant. This is
    forced by what 9.5.5 says the variants do. G1 is meant to give the nut d of
    room below it, and it achieves that by dropping the table under the well --
    the nut does not move. Re-seating the nut on each variant would put it on the
    well floor and report cl_+ = 0 there too, which is the reading under which G1
    changes nothing and T8 has no independent variable.

    x* itself is the *seated* pose, not the axis-coincidence pose that
    `compute_task_tol` uses. Those differ along the insertion axis by however far
    the part has yet to fall, and the axial offset is exactly what cl_pm measures,
    so taking the tolerance script's pose would report the drop height instead of
    the clearance. Seating is a lift followed by the same stepped query: the part
    is put on the hole axis at the terminal yaw, raised along -v until the two
    bodies' extents no longer overlap, and then dropped along +v to first contact.

    The lift is not a formality. `place_at_terminal_pose` puts the moving body's
    origin *on* the static body's, which is the right thing for a cross-section
    clearance and leaves the two solids interpenetrating for anything axial. On
    `Stack` that is a complete overlap of the two cubes, and a drop from there
    reports zero in both directions -- a pose in contact with everything has no
    clearance to measure.
    """
    from portal_trace.registry.geometry import (
        CLEARANCE_STEP_M,
        TerminalPose,
        body_extent_along,
        make_env,
    )
    from portal_trace.registry.geometry import measure_clearance as sweep
    from portal_trace.scripts.compute_task_tol import place_at_terminal_pose

    spec = task(task_id)
    if spec.moving_body is None or spec.insertion_axis is None:
        raise ValueError(
            f"{task_id} declares no moving body or insertion axis, so it has no "
            "terminal compensation direction and cl_pm is undefined on it"
        )
    known = tuple(v.name for v in spec.geom_variants)
    names = tuple(variants) if variants else known
    if unknown := set(names) - set(known):
        raise KeyError(f"{task_id} has no geometry variant {sorted(unknown)}; it has {known}")
    # Registry order, not the order the caller listed them in.
    names = tuple(n for n in known if n in set(names))
    if "G0" not in names:
        raise ValueError(
            "x* is seated on G0 and reused, and G2's baffle is placed relative to "
            "the nut's height there, so G0 has to be in the same call. Measuring a "
            "variant against a pose seated on itself answers a different question"
        )

    v = np.asarray(spec.insertion_axis, dtype=np.float64)
    v = v / np.linalg.norm(v)
    x_star: TerminalPose | None = None
    seating: dict[str, Any] = {}
    out: dict[str, Any] = {}

    for name in names:
        env = make_env(
            spec,
            family,
            variant=name,
            well_depth=well_depth,
            baffle_height=baffle_height,
            nut_terminal_z_m=None if x_star is None else float(x_star.position[2]),
        )
        try:
            env.reset()
            if x_star is None:
                model, data = env.sim.model._model, env.sim.data._data
                place_at_terminal_pose(model, data, spec)
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.moving_body)
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[body_id], dtype=np.float64))
                coincident = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
                static_name = (
                    spec.hole_body
                    if spec.moving_body == spec.inserted_body
                    else spec.inserted_body
                )
                assert static_name is not None
                moving_lo, moving_hi = body_extent_along(model, data, spec.moving_body, v)
                static_lo, _ = body_extent_along(model, data, static_name, v)
                # Along +v the moving body has to end up entirely before the static
                # one begins. One extra step of the sweep so the lifted pose is
                # strictly clear rather than exactly touching.
                lift = max(0.0, moving_hi - static_lo) + CLEARANCE_STEP_M
                approach = TerminalPose(
                    body=spec.moving_body,
                    position=coincident - lift * v,
                    quaternion=quat,
                )
                drop = sweep(env, approach, v)
                if drop.contact_at_pose:
                    raise RuntimeError(
                        f"{task_id}: lifting {spec.moving_body} by "
                        f"{lift * 1e3:.2f} mm left it in contact, so its extent along "
                        "the insertion axis is under-reported and the seated pose "
                        "cannot be reached from above"
                    )
                if drop.saturated_plus:
                    raise RuntimeError(
                        f"{task_id}: the {spec.moving_body} fell the whole "
                        f"{drop.max_steps * drop.step_m * 1e3:.0f} mm of the sweep "
                        "without touching anything, so it has no seated pose. The "
                        "axis-coincidence placement is not over the receptacle"
                    )
                x_star = TerminalPose(
                    body=approach.body,
                    position=approach.position + drop.cl_plus * v,
                    quaternion=approach.quaternion,
                )
                seating = {
                    "coincident_position": coincident.tolist(),
                    "moving_extent_along_v": [moving_lo, moving_hi],
                    "static_extent_along_v": static_lo,
                    "lift_m": lift,
                    "drop_m": drop.cl_plus,
                    "seated_position": x_star.position.tolist(),
                    "seated_against": list(drop.first_contact_plus or ()),
                }
            result = sweep(env, x_star, v)
        finally:
            env.close()
        out[name] = result.to_json()
    out["_seating"] = seating
    return out


def measure_gzip_ratio(task_id: str, family: str, *, n_steps: int) -> dict[str, float]:
    """Observed gzip ratio per modality, from real rollout data.

    9.8.3's 758 GB rests on this. Measured on actual frames rather than assumed,
    because the RGB ratio in particular depends on how much of the 84x84 frame is
    flat background, which is a property of these scenes.
    """
    cameras = ("agentview", "robot0_eye_in_hand")
    env = _make_env(task(task_id), family, cameras=cameras)
    try:
        env.reset()
        images: list[np.ndarray] = []
        lowdim: list[np.ndarray] = []
        action = np.zeros(env.action_dim)
        for _ in range(n_steps):
            obs, *_ = env.step(action)
            images.append(np.stack([obs[f"{c}_image"] for c in cameras]))
            lowdim.append(np.asarray(obs["robot0_proprio-state"], dtype=np.float32))
    finally:
        env.close()

    def ratio(arr: np.ndarray) -> float:
        raw = arr.tobytes()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=4, mtime=0) as fh:
            fh.write(raw)
        return len(raw) / len(buf.getvalue())

    return {
        "rgb84": ratio(np.stack(images)),
        "lowdim_float32": ratio(np.stack(lowdim)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_WHATS = (
    "sim_facts",
    "tau_step",
    "tau_frame",
    "train_throughput",
    "proprio_dim",
    "armature",
    "actuator_limits",
    "hard_reset",
    "camera_pose",
    "gzip_ratio",
    "d7_stability",
    "clearance",
    "stiction",
)


def _report(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, dict):
        for k, v in payload.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {payload}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--what", nargs="+", choices=(*_WHATS, "all"), default=["all"])
    p.add_argument("--task", default="NutAssemblySquare", choices=sorted(TASKS))
    p.add_argument("--family", default=_DEFAULT_FAMILY)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument(
        "--formal-plan",
        type=Path,
        default=None,
        help="hash-bound formal data plan required by --what train_throughput",
    )
    p.add_argument(
        "--formal-normalization",
        type=Path,
        default=None,
        help="matching train-only normalization artifact required by train_throughput",
    )
    p.add_argument("--train-arm", default="B2-naive-pooling")
    p.add_argument("--train-architecture", default="resnet18_transformer4")
    p.add_argument("--train-device", default="cuda")
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=20)
    p.add_argument("--train-warmup", type=int, default=5)
    p.add_argument("--train-d-model", type=int, default=512)
    p.add_argument("--train-layers", type=int, default=4)
    p.add_argument("--train-heads", type=int, default=8)
    p.add_argument("--train-ffn-dim", type=int, default=2048)
    p.add_argument("--train-no-augment", action="store_true")
    p.add_argument("--d7-steps", type=int, default=40)
    p.add_argument("--d7-grid", type=int, default=16)
    p.add_argument(
        "--clearance-variants",
        nargs="*",
        default=None,
        help="default: every geometry variant the task declares",
    )
    p.add_argument(
        "--well-depth-m",
        type=float,
        default=None,
        help="G1/G2's d. 9.5.5 has no default for it: P0 fixes it from F8's "
        "feasibility check after cl_pm and ||c_delta(x*)|| are known",
    )
    p.add_argument("--baffle-height-m", type=float, default=None, help="G2's h, likewise")
    p.add_argument("--stiction-grid", type=int, default=80)
    p.add_argument("--stiction-hold", type=float, default=0.5)
    p.add_argument("--write-lock", action="store_true")
    args = p.parse_args(argv)

    check_pinned_versions(required=("mujoco", "robosuite"))
    requested_all = "all" in args.what
    wants = set(_WHATS) if requested_all else set(args.what)
    # Throughput needs a real materialised policy plan and is intentionally not
    # part of the legacy no-argument simulator sweep.  It remains available in
    # ``all`` when those inputs are supplied, but explicit omission is visible.
    if requested_all and (
        args.formal_plan is None or args.formal_normalization is None
    ):
        wants.discard("train_throughput")
        print(
            "[skip] train_throughput: pass --formal-plan and --formal-normalization "
            "to measure the real policy stack"
        )
    commit = source_commit()
    lock = LockFile() if args.write_lock else None
    results: dict[str, Any] = {}

    if "sim_facts" in wants:
        results["sim_facts"] = measure_simulation_facts(args.task, args.family)
        _report("simulation facts", results["sim_facts"])

    if "hard_reset" in wants:
        results["hard_reset"] = measure_hard_reset_default()
        _report("robosuite defaults", results["hard_reset"])
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["sim.hard_reset_default"],
                results["hard_reset"]["hard_reset"],
                source_commit=commit,
                notes=f"lite_physics default is {results['hard_reset']['lite_physics']}",
            )

    if "proprio_dim" in wants:
        results["proprio_dim"] = measure_proprio_dim(args.family)
        _report("proprio layout", results["proprio_dim"])
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["obs.proprio_native_dim.{family}"],
                results["proprio_dim"]["native"],
                source_commit=commit,
                notes=results["proprio_dim"]["breakdown"],
                extra=results["proprio_dim"],
                family=args.family,
            )

    if "armature" in wants:
        results["armature"] = measure_armature(args.family)
        _report("per-joint dof properties", results["armature"])
        if lock:
            decl = REQUIRED_MEASUREMENTS["mjcf.armature.{family}.{joint}"]
            for joint, vals in results["armature"].items():
                lock.put(
                    decl,
                    vals["armature"],
                    source_commit=commit,
                    notes=f"damping={vals['damping']}, frictionloss={vals['frictionloss']}",
                    extra=vals,
                    family=args.family,
                    joint=joint,
                )

    if "actuator_limits" in wants:
        results["actuator_limits"] = measure_actuator_limits(args.family)
        _report("actuator limits", {"n": len(results["actuator_limits"]["actuators"])})
        for a in results["actuator_limits"]["actuators"]:
            print(f"    {a}")

    if "d7_stability" in wants:
        d7 = measure_d7_stability(
            args.task, args.family, n_steps=args.d7_steps, n_grid=args.d7_grid
        )
        results["d7_stability"] = d7
        _report("D7 integrator limit", {k: v for k, v in d7.items() if k != "trials"})
        for t in d7["trials"]:
            print(f"    N={t['n_over_defective']:2d}  omega*dt={t['omega_dt']:.4f}  "
                  f"ratio={t['link_compliance']:.6g}  "
                  f"{'DIVERGED' if t['diverged'] else 'stable'} "
                  f"after {t['steps_survived']} steps")
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["defect.d7_clean_compliance"],
                d7["link_compliance"],
                source_commit=commit,
                notes=(
                    f"rigid twin at N={d7['n_over_defective']}x the defective mode, "
                    f"omega*dt={d7['omega_dt_clean']:.4f}; last stable "
                    f"{d7['omega_dt_last_stable']:.4f}, first divergent "
                    f"{d7['omega_dt_first_divergent']:.4f}; margin {d7['margin']}; "
                    f"swept on {args.family}/{args.task}, all {len(d7['trials'])} grid "
                    f"points run to {d7['n_steps']} control steps"
                ),
                extra=d7,
            )

    if "stiction" in wants:
        stiction = measure_stiction(
            args.family, n_grid=args.stiction_grid, hold_time_s=args.stiction_hold
        )
        results["stiction"] = stiction
        print(f"\n=== D6 stiction offset, {args.family} ===")
        print(
            f"  {'joint':20s} {'f_bar':>8s} {'constraint':>11s} {'force':>8s} "
            f"{'offset':>8s}  verdict"
        )
        for joint, r in stiction.items():
            print(
                f"  {joint:20s} {r['f_bar']:8.4f} {r['threshold_constraint']:11.4f} "
                f"{r['threshold_force']:8.4f} {r['offset']:8.4f}  "
                f"{'REJECTS D6' if r['rejects_d6'] else 'within 0.025'}"
            )
        worst = max(stiction.values(), key=lambda r: abs(r["offset"]))
        print(
            f"\n  worst |offset| = {abs(worst['offset']):.4f} N m against the "
            f"{STICTION_REJECT_ABOVE:.4f} N m bound of 9.2.6: D6 "
            f"{'LEAVES' if any(r['rejects_d6'] for r in stiction.values()) else 'stays in'}"
            " the 36 twin pairs"
        )
        if lock:
            decl = REQUIRED_MEASUREMENTS["defect.d6_stiction_offset.{family}.{joint}"]
            for joint, r in stiction.items():
                lock.put(
                    decl,
                    r["offset"],
                    source_commit=commit,
                    notes=(
                        f"constraint {r['threshold_constraint']:.4f} - force "
                        f"{r['threshold_force']:.4f} N m, f_bar = {r['f_bar']:.4f}, "
                        f"ladder step {r['ladder_step']:.5f} N m, "
                        f"{r['hold_time_s']} s hold; "
                        f"{'rejects' if r['rejects_d6'] else 'within'} the 0.025 bound"
                    ),
                    extra={k: v for k, v in r.items() if k != "trace"},
                    family=args.family,
                    joint=joint,
                )

    if "clearance" in wants:
        variants = tuple(args.clearance_variants) if args.clearance_variants else None
        clearance = measure_clearance(
            args.task,
            args.family,
            variants=variants,
            well_depth=args.well_depth_m,
            baffle_height=args.baffle_height_m,
        )
        results["clearance"] = clearance
        seating = clearance.pop("_seating")
        print(
            f"\n=== x* {args.task} ===\n"
            f"  lifted {seating['lift_m'] * 1e3:.3f} mm clear, then dropped "
            f"{seating['drop_m'] * 1e3:.3f} mm onto {seating['seated_against']}"
        )
        for name, r in clearance.items():
            print(
                f"\n=== cl_pm {args.task} {name} ===\n"
                f"  cl_+ = {r['cl_plus_m'] * 1e3:8.3f} mm"
                f"{'  (SATURATED)' if r['saturated_plus'] else ''}\n"
                f"  cl_- = {r['cl_minus_m'] * 1e3:8.3f} mm"
                f"{'  (SATURATED)' if r['saturated_minus'] else ''}\n"
                f"  A    = {r['asymmetry_m'] * 1e3:8.3f} mm   sign = "
                f"{'+' if r['asymmetry_m'] > 0 else '-'}\n"
                f"  first contact +/-: {r['first_contact_plus']} / {r['first_contact_minus']}\n"
                f"  in contact at x*: {r['contact_at_pose']}"
            )
        if lock:
            decl = REQUIRED_MEASUREMENTS["geom.clearance.{task}.{geom_variant}.{direction}"]
            for name, r in clearance.items():
                for direction, key in (("plus", "cl_plus_m"), ("minus", "cl_minus_m")):
                    lock.put(
                        decl,
                        r[key],
                        source_commit=commit,
                        notes=(
                            f"stepped mj_collision at {r['step_m'] * 1e3:.2f} mm over at "
                            f"most {r['max_steps']} steps, on {args.family}; "
                            f"A = {r['asymmetry_m'] * 1e3:.3f} mm"
                        ),
                        extra=r,
                        task=args.task,
                        geom_variant=name,
                        direction=direction,
                    )

    if "camera_pose" in wants:
        results["camera_pose"] = measure_camera_pose(args.task, args.family)
        _report("camera poses", results["camera_pose"])
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["mjcf.agentview_pose.{task}"],
                results["camera_pose"],
                source_commit=commit,
                task=args.task,
            )

    if "tau_step" in wants:
        t = measure_tau_step(args.task, args.family, n_steps=args.n_steps, n_warmup=args.n_warmup)
        results["tau_step"] = t
        _report("tau_step", {"mean_s_per_step": t.seconds_per_call,
                             "median_s_per_step": t.median_seconds,
                             "std": t.std_seconds, "n": t.n_calls})
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["sim.tau_step.{family}.{task}"],
                t.seconds_per_call,
                source_commit=commit,
                notes=f"mean of {t.n_calls} steps after {t.n_warmup} warm-up; "
                      f"median {t.median_seconds:.4g} s, std {t.std_seconds:.4g} s, "
                      "render off",
                family=args.family,
                task=args.task,
            )

    if "tau_frame" in wants:
        t = measure_tau_frame(args.task, args.family, n_steps=args.n_steps, n_warmup=args.n_warmup)
        results["tau_frame"] = t
        _report("tau_frame (marginal, per camera)",
                {"mean_s_per_frame": t.seconds_per_call,
                 "median_s_per_frame": t.median_seconds, "std": t.std_seconds})
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["sim.tau_frame.{family}.{task}"],
                t.seconds_per_call,
                source_commit=commit,
                notes=f"marginal cost of one rendered 84x84 frame, two cameras on; "
                      f"median {t.median_seconds:.4g} s",
                family=args.family,
                task=args.task,
            )

    if "train_throughput" in wants:
        if args.formal_plan is None or args.formal_normalization is None:
            raise ValueError(
                "--what train_throughput requires --formal-plan and "
                "--formal-normalization; synthetic or guessed throughput is not admissible"
            )
        from portal_trace.policies.architecture import canonical_architecture
        from portal_trace.policies.interface import PolicyConfig
        from portal_trace.runners.formal_adapter import measure_formal_adapter_throughput
        from portal_trace.runners.formal_data import (
            load_formal_data_plan,
            load_formal_normalization,
        )

        plan = load_formal_data_plan(args.formal_plan)
        normalization = load_formal_normalization(args.formal_normalization, plan=plan)
        architecture = canonical_architecture(args.train_architecture)
        if architecture != "resnet18_transformer4":
            raise ValueError(
                "train_throughput currently implements only resnet18_transformer4; "
                f"refusing to measure or publish {architecture!r} until its policy "
                "adapter is implemented"
            )
        config = PolicyConfig(
            arm=args.train_arm,
            d_model=args.train_d_model,
            n_layers=args.train_layers,
            n_heads=args.train_heads,
            ffn_dim=args.train_ffn_dim,
            augment=not args.train_no_augment,
        )
        measured = measure_formal_adapter_throughput(
            plan,
            normalization,
            config,
            seed=0,
            batch_size=args.train_batch_size,
            n_steps=args.train_steps,
            n_warmup=args.train_warmup,
            device=args.train_device,
        )
        measured["architecture"] = architecture
        results["train_throughput"] = measured
        _report("formal adapter throughput", measured)
        if lock:
            lock.put(
                REQUIRED_MEASUREMENTS["train.throughput.{arch}"],
                measured["steps_per_second"],
                source_commit=commit,
                notes=(
                    f"{measured['n_steps']} timed optimizer steps after "
                    f"{measured['n_warmup']} warm-up steps on {measured['device']}; "
                    "plan/config hashes are recorded in extra"
                ),
                extra=measured,
                arch=architecture,
            )

    if "gzip_ratio" in wants:
        results["gzip_ratio"] = measure_gzip_ratio(
            args.task, args.family, n_steps=min(args.n_steps, 100)
        )
        _report("gzip ratios", results["gzip_ratio"])
        if lock:
            decl = REQUIRED_MEASUREMENTS["storage.gzip_ratio.{modality}"]
            for modality, r in results["gzip_ratio"].items():
                lock.put(decl, r, source_commit=commit, modality=modality)

    if lock:
        lock.set_versions({k: v or "absent" for k, v in fingerprint(Backend.CPU_MUJOCO)
                           .versions.items()})
        lock.set_backend(Backend.CPU_MUJOCO.value)
        lock.save()
        print(f"\n[lock] {lock.path}")
        outstanding = lock.outstanding()
        print(f"[lock] {len(outstanding)} declarations still outstanding:")
        for decl in outstanding:
            print(f"    {decl.key_template:44s} <- {decl.produced_by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
