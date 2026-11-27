"""`a_dot`: the closed-loop vector field derivative on the clean twin.

Section 5.1 fixes the definition:

    a_dot(x) := grad_x pi_{e,0,tau}(x) . f_{e_0}(x, pi_{e,0,tau}(x))

and the reason it is written this way rather than as a time derivative along the
reference trajectory: the primary endpoints are evaluated on rho^off, whose
states lie on no reference trajectory, and a time derivative is undefined there.
The form above is defined on all of X and is computable at any x, because (A2)
grants off-support queries on the clean twin -- one forward step gives f_{e_0}
and a finite difference gives the policy Jacobian.

This is D2's ground truth: c = ell * a_dot, with ell = n / f_p. Appendix A-23
forbids the torque-side caliber, and the reason is exactly that this closed form
stops being true there -- a torque delay sits inside the OSC feedback loop, so
the steady-state lag is set by the loop dynamics rather than by ell, and the high
levels show up as oscillation instead of the undertravel Proposition 2(c)
classifies D2 as. `caliber` therefore has no default: a caller has to name it,
and naming the torque side raises.

Why the policy protocol carries a control step
----------------------------------------------

5.1 writes pi_{e,0,tau}(x) with one argument, and for a stationary feedback
policy the boxed form above *is* the rate of change of the action the policy
emits as it is carried along the closed loop. The teacher of 9.4.2 is not
stationary: its action is OSC(x; x_ref[t]) + u_inf[t], so it depends on the
control step through both the reference waypoint and the ILC feedforward, and
9.6.7 states the consequence outright -- "the signature is query_clean(x, t) and
not query_clean(x), because u_inf's phase can only come from the control step
index", and every estimator of a_dot has to be written on two arguments.

For such a teacher the boxed form splits into a convective part and an explicit
part, and only their sum is the object 9.2.2 derives the closed form from. That
derivation is: with the command stream delayed by ell, the teacher must emit at
t what it would otherwise emit at t + ell, so

    c[t] = a^nom[t + ell] - a^nom[t] = ell * d/dt a^nom + O(ell^2),

a total derivative of the emitted action sequence along the clean closed loop.
Dropping the explicit part would keep only -grad_x of the tracker's own pose
feedback, which is the end-effector velocity with a minus sign -- roughly the
negation of the term the reference contributes -- so it is not a small omission
but a different vector, and it would be compared against the ILC feedforward
under 9.4.7's pre-registered cosine floor. On Stack / Panda / D2 at 50 ms the
two parts have norms 21.1 and 21.5 against a total of 6.1 and lie at cos -0.96
to each other, and against the ILC compensation the total reads cos 0.90 while
the convective part alone reads 0.04: the choice is between the quantity and
noise, not between two calibers of it. `DerivativeSplit` reports the two parts
separately so that this stays a measurement rather than the argument above.

The alternative caliber 5.1 does demote to a sensitivity control is a different
object again: projecting the reference trajectory's time derivative onto the
state feature basis phi(x). That one is estimated in `teacher.project`, and the
form here needs no estimator at all.

What a one-step push-forward has to be given
--------------------------------------------

5.1 says the push-forward "is `mj_step`", and `physics_state` is (qpos, qvel), so
the natural reading is that those two plus the action determine x_{t+1}. They do
not. Replaying a recorded frame of the 9.4.7 cell (Stack / Panda / D2 level 1)
through `flow` reproduces x_{t+1} to 5e-16 in free space and misses `qvel` by
up to 0.40 rad/s while the hand is on the cube. Restoring one candidate carrier
at a time, at three grasp frames -- the twin left wherever the query before it
put it, which is a spot query and is the case the instrument exists for --
isolates which:

    restored on top of (qpos, qvel)      frame 120   frame 200   frame 300
    ---------------------------------------------------------------------
    nothing                               1.14e-01    4.01e-01    3.96e-01
    qacc_warmstart                        1.14e-01    4.01e-01    3.96e-01
    mjData.ctrl                           1.14e-01    4.01e-01    3.96e-01
    mjData.qfrc_applied                   1.14e-01    4.01e-01    3.96e-01
    mjData.time                           1.14e-01    4.01e-01    3.96e-01
    mjData.act (empty on this model)      1.14e-01    4.01e-01    3.96e-01
    OSC goal_pos, goal_ori                1.14e-01    4.01e-01    3.96e-01
    the wrapper's own defect_state        1.14e-01    4.01e-01    3.96e-01
    the whole mjSTATE_INTEGRATION         1.14e-01    4.01e-01    3.96e-01
    robosuite's gripper accumulator       2.1e-12     2.0e-14     2.1e-14
    that and qacc_warmstart               0           0           0

The carrier is therefore not MuJoCo's -- restoring everything `mj_getState`
knows about changes nothing -- it is robosuite's. `GripperModel.format_action`
integrates sign(a_gripper) at 0.2 per control step and clips to [-1, 1], so the
finger setpoint is a saturating integrator of the command and lives in a Python
attribute. `flow` takes it as a required argument, for the same reason
`query.TeacherArm` refuses to answer without z: an off-support x has no
accumulator of its own and there is no value to assume for one. The OSC goal
contributes nothing because 9.4.2 runs it in `goal_update_mode="achieved"`, where
`set_goal` recomputes the goal from the measured pose every control step; under
`"desired"` it would integrate and would have to be carried too.

Why the error was invisible where it mattered most
--------------------------------------------------

The size of the miss is set by how far the twin's accumulator is from the query
frame's, so it is a function of the query *order*, and one order hides it
completely. Sweeping t = 0 .. T-2 in ascending order without passing the
accumulator at all reproduces every contact frame to 4.1e-10: the accumulator the
query at t-1 leaves behind is, exactly, the one the rollout had at t, because both
ran `format_action` once on the same command. The same 399 frames swept backwards
miss by 4.2e-01. That is the whole pre-fix behaviour -- not an approximation with
a small constant, but a bug whose magnitude is zero on the one access pattern the
code happened to be exercised on and 0.4 rad/s off it. So it costs 9.4.7's cosine
almost nothing (0.899307 either way, the batch there being an ascending sweep) and
would have cost an arbitrary amount as soon as anything queried rho^off, which is
what 5.1 grants (A2) for. It is recorded here at that size rather than at the size
it currently shows, because the second number is a property of the call site.

`qacc_warmstart` is the second one and is a different kind of object -- the
constraint solver's initial guess, not state. It cannot be an argument, because
an off-support x has no recorded guess either, so `_set_state` fixes it at zero
and `flow` is a function of (x, a) by construction: two query orders over the
same frames agree bit for bit, which is not true of any policy that leaves the
previous query's value in place. The price is that `flow` no longer reproduces a
recorded contact transition exactly -- the rollout's own step was seeded with the
previous step's acceleration and MuJoCo's solver does not run to machine
precision -- and the residual is 1.7e-8 rad/s of `qvel`, worst frame of the 399,
contact only, against a `qvel` scale of 0.66. Free-space frames are unaffected.

Two readings of `qacc_warmstart` are worth separating, because the first one
looks like a fix and is not. Writing the recorded value *before* `sim.forward()`
does nothing: `mj_forward` re-solves and overwrites it. Writing it *after* makes
the reproduction exact, and is what the table's last row does -- but it is only
available at a frame that was recorded, which is not the case the instrument
exists for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from portal_trace.registry.tasks import CONTROL_FREQ

__all__ = [
    "BodySnapshot",
    "DelayCaliber",
    "DerivativeMethod",
    "ForbiddenCaliber",
    "ClosedLoopDerivative",
    "DerivativeReport",
    "DerivativeSplit",
    "Policy",
    "assert_action_side_caliber",
    "delay_seconds",
]

# Everything `mj_setState` will put back: qpos, qvel, act, time, plugin state,
# the warm start, and the user inputs `ctrl` / `qfrc_applied` / `xfrc_applied` /
# `eq_active` / mocap / userdata. Named once so that a snapshot cannot drift into
# being a subset of it.
_INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION

# (state, control step) -> action. The second argument is not optional and there
# is no one-argument overload: a teacher that ignores it can close over nothing
# and drop it, while a teacher that needs it and is offered the one-argument
# protocol would silently return the convective part alone (see the module
# docstring).
Policy = Callable[[np.ndarray, int], np.ndarray]


class DelayCaliber(str, Enum):
    """Where D2's delay is inserted."""

    ACTION_SIDE = "action_side"    # the control-step buffer of 9.2.2, the only one allowed
    TORQUE_SIDE = "torque_side"    # inside the OSC loop, excluded by A-23


class DerivativeMethod(str, Enum):
    """Two ways to evaluate the same object; neither is a fallback for the other.

    `FLOW` differences the policy between (x, t) and (flow(x), t + 1), two
    evaluations and one step. `JACOBIAN` builds grad_x pi column by column,
    contracts it with the measured f and adds the explicit phase derivative,
    costing 2 n_x + 2 evaluations. They agree to O(dt) -- about 2% of the norm on
    the 9.4.2 teacher -- and `cross_check` reports the gap, which is a diagnostic
    on the policy's smoothness rather than a choice between them.
    """

    FLOW = "flow"
    JACOBIAN = "jacobian"


class ForbiddenCaliber(ValueError):
    """An excluded implementation path was requested (appendix A)."""


def assert_action_side_caliber(caliber: DelayCaliber) -> None:
    if DelayCaliber(caliber) is not DelayCaliber.ACTION_SIDE:
        raise ForbiddenCaliber(
            "appendix A-23 excludes the torque-side delay caliber. It lands inside "
            "the OSC feedback loop, so the steady-state tracking lag is set by the "
            "loop dynamics rather than by ell and c = ell * a_dot -- the closed form "
            "that carries Lambda's numerator and denominator, the surrogate "
            "amplitude match, kappa's numerator, E4's ground-truth cosine and B8's "
            "a - c_true -- is simply false. It is also marginally stable: at "
            "kp = 150, zeta = 1 and about 2 kg of end-effector inertia the loop sits "
            "at 1.4 Hz, where 100 ms is 50 degrees of phase. Use the control-step "
            "action-side buffer of 9.2.2."
        )


def delay_seconds(n_control_steps: float) -> float:
    """ell = n / f_p. Integer levels only, so the zero-order hold is exact (9.2.2)."""
    n = float(n_control_steps)
    if n != round(n):
        raise ValueError(
            f"{n} is not an integer number of control steps. A fractional shift needs "
            "first-order hold interpolation, which is a low-pass filter coupled to "
            "ell: it mixes an amplitude roll-off into D2's effect and breaks the "
            "all-pass property that Proposition 2(c) rests on (9.2.2 reason 1)."
        )
    return n / CONTROL_FREQ


@dataclass(frozen=True)
class BodySnapshot:
    """Where a body was before a batch of off-support queries moved it.

    `integration` is MuJoCo's own `mjSTATE_INTEGRATION`, taken whole rather than
    field by field: it is the widest thing `mj_setState` will restore, and an
    enumeration here would have to be revisited every time the step loop touches
    another array -- `flow` writes `ctrl` and `qfrc_applied` through
    `DefectWrapper._pre_action`, which a (qpos, qvel, time) snapshot silently left
    shifted. `warmstart` is a copy of the slice `mjSTATE_INTEGRATION` already
    carries, because restoring it has to happen after the forward solve that
    `_restore` runs and that solve overwrites it.

    The other two are the carriers MuJoCo does not know about: robosuite's gripper
    command accumulator and the wrapper's own memory state.
    """

    integration: np.ndarray
    warmstart: np.ndarray
    actuation: np.ndarray
    defect_state: np.ndarray


@dataclass(frozen=True)
class DerivativeReport:
    """The gap between the two calibers of `DerivativeMethod`."""

    max_abs_difference: float
    relative_difference: float
    n_states: int


@dataclass(frozen=True)
class DerivativeSplit:
    """`a_dot` resolved into the two terms a time-varying teacher contributes.

    `convective` is grad_x a . f, the term 5.1's boxed form writes; `explicit` is
    d/dt a at frozen state, which exists because the reference waypoint and the
    ILC feedforward both move with the control step. `total` is the derivative
    along the closed loop and is what D2's closed form multiplies by ell; it is
    measured directly rather than added up, so `cross` -- the difference between
    it and the sum of the two parts -- is the second-order term of the two-way
    difference and not a rounding of it.
    """

    convective: np.ndarray
    explicit: np.ndarray
    total: np.ndarray

    @property
    def cross(self) -> np.ndarray:
        return self.total - self.convective - self.explicit

    @property
    def explicit_share(self) -> float:
        """||explicit|| / ||total||: how much of D2's field the term carries."""
        scale = float(np.linalg.norm(self.total))
        if scale == 0.0:
            raise ValueError(
                "a_dot is identically zero over these frames, so the explicit "
                "term has no share to report"
            )
        return float(np.linalg.norm(self.explicit)) / scale

    def render(self) -> str:
        return (
            f"a_dot over {len(self.total)} frames: ||grad_x a . f|| "
            f"{np.linalg.norm(self.convective):.4f}, ||da/dt|| "
            f"{np.linalg.norm(self.explicit):.4f}, ||total|| "
            f"{np.linalg.norm(self.total):.4f}, cross term "
            f"{np.linalg.norm(self.cross):.4f}"
        )


class ClosedLoopDerivative:
    """Evaluates `a_dot` on a clean twin.

    The twin is a `DefectWrapper` carrying `delta.zero()`; it is stepped through
    its own `_pre_action`, so the derivative is taken through exactly the code
    path the rollouts use, rather than through a second, simplified model of it.
    Every query saves and restores the MuJoCo state, the solver's warm start, the
    gripper accumulator and the wrapper's defect state -- the complete carrier set
    of the module docstring -- so a sequence of off-support queries leaves the
    twin where it started.

    `policy` is the clean teacher, on the two-argument protocol above. It is
    supplied rather than constructed here because binding a reference trajectory
    and an ILC feedforward to a body is `teacher.query`'s job and this module
    sits below it; `teacher.query.a_dot_nominal` is the adapter that does it.
    """

    def __init__(self, clean_twin: Any, policy: Policy, *, fd_step: float = 1.0e-6) -> None:
        self.twin = clean_twin
        self.policy = policy
        self.fd_step = float(fd_step)
        if self.fd_step <= 0.0:
            raise ValueError(f"finite-difference step {fd_step} is not positive")
        if not clean_twin.delta.is_zero:
            raise ValueError(
                f"a_dot is defined on the clean twin (5.1); this wrapper carries "
                f"{clean_twin.delta}"
            )
        self.dt = 1.0 / CONTROL_FREQ
        raw = clean_twin.env.sim.model._model
        self.nq = int(raw.nq)
        self.nv = int(raw.nv)

    # -- primitives ------------------------------------------------------

    def _snapshot(self) -> BodySnapshot:
        model, data = self._model_data()
        buf = np.empty(mujoco.mj_stateSize(model, _INTEGRATION))
        mujoco.mj_getState(model, data, buf, _INTEGRATION)
        return BodySnapshot(
            integration=buf,
            warmstart=np.array(data.qacc_warmstart),
            actuation=self.twin.actuation_state(),
            defect_state=self.twin.defect_state(),
        )

    def _restore(self, snapshot: BodySnapshot) -> None:
        model, data = self._model_data()
        mujoco.mj_setState(model, data, snapshot.integration, _INTEGRATION)
        # the forward pass leaves the derived quantities -- site poses, Jacobians,
        # the mass matrix -- consistent with the restored qpos, which a caller
        # reading the body straight after a query batch depends on. It also
        # re-solves the constraints and writes its own warm start, so the recorded
        # one goes back afterwards.
        self.twin.env.sim.forward()
        data.qacc_warmstart[:] = snapshot.warmstart
        self.twin.set_actuation_state(snapshot.actuation)
        self.twin.set_defect_state(snapshot.defect_state)

    def _model_data(self) -> tuple[Any, Any]:
        """The raw MuJoCo handles, re-read every time.

        `hard_reset` defaults to `True` and rebuilds `MjSim` from the MJCF, so a
        handle cached in `__init__` would outlive the simulator it points at and
        read a frozen copy of the state.
        """
        sim = self.twin.env.sim
        return sim.model._model, sim.data._data

    def _set_state(self, state: np.ndarray) -> None:
        """Write x and leave the solver's warm start at the one value x implies.

        Zero rather than whatever the last query left, and zero rather than the
        forward solve's own output: the module docstring measures both, and only
        the fixed value makes `flow` a function of (x, a) instead of a function of
        the query order as well.
        """
        if state.size != self.nq + self.nv:
            raise ValueError(
                f"a state is nq + nv = {self.nq + self.nv} long, got {state.size}"
            )
        sim = self.twin.env.sim
        sim.data.qpos[:] = state[: self.nq]
        sim.data.qvel[:] = state[self.nq :]
        sim.forward()
        sim.data._data.qacc_warmstart[:] = 0.0

    def flow(
        self, state: np.ndarray, action: np.ndarray, *, actuation: np.ndarray
    ) -> np.ndarray:
        """One control step of the clean twin's closed loop, from x under a.

        This is f_{e_0} integrated over dt. The substep loop is robosuite's own,
        minus the observable and reward bookkeeping, so that an off-support query
        does not advance the episode counter or trip the horizon.

        `actuation` is the gripper command accumulator at the query point, from
        `DefectWrapper.actuation_state()`. It has no default and no None branch:
        the accumulator is a saturating integrator of the gripper command that no
        part of `state` holds, and letting the previous query's value stand is
        worth up to 0.40 rad/s of `qvel` at a grasp frame (module docstring).
        """
        sim = self.twin.env.sim
        self._set_state(state)
        self.twin.set_actuation_state(actuation)
        n_substeps = int(round(self.twin.env.control_timestep / self.twin.env.model_timestep))
        for i in range(n_substeps):
            sim.step1()
            self.twin._pre_action(action, i == 0)
            sim.step2()
        data = sim.data._data
        return np.concatenate([np.array(data.qpos), np.array(data.qvel)])

    # -- the derivative ---------------------------------------------------

    def _frames(
        self, states: np.ndarray, steps: np.ndarray, actuation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate one batch of (x, t, actuation) query points.

        The pairing is checked here rather than left to the policy because a
        broadcast that happened to succeed would evaluate the whole batch at one
        phase, which is the frozen-phase quantity the module docstring rules out
        and which no downstream shape check could catch. The accumulator is
        checked in the same place and for the same reason: one row broadcast over
        the batch would put one gripper configuration on every frame, which is the
        error the module docstring measures at 0.40 rad/s.
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        steps = np.atleast_1d(np.asarray(steps, dtype=np.int64))
        actuation = np.atleast_2d(np.asarray(actuation, dtype=np.float64))
        if steps.shape != (states.shape[0],):
            raise ValueError(
                f"{states.shape[0]} states and {steps.shape[0]} control-step "
                "indices. a_dot is a two-argument object (9.6.7) and every frame "
                "carries its own phase"
            )
        if actuation.shape[0] != states.shape[0]:
            raise ValueError(
                f"{states.shape[0]} states and {actuation.shape[0]} rows of "
                "gripper accumulator. The accumulator is per frame -- it is a "
                "saturating integrator of the gripper command, so one row is one "
                "hand configuration and not a property of the episode"
            )
        return states, steps, actuation

    def a_dot(
        self,
        states: np.ndarray,
        steps: np.ndarray,
        *,
        caliber: DelayCaliber,
        actuation: np.ndarray,
        method: DerivativeMethod = DerivativeMethod.FLOW,
    ) -> np.ndarray:
        """(N, nq + nv) states and (N,) control steps in, (N, action_dim) out.

        The step at frame i is the phase the teacher is queried at; the frame is
        differenced against step i + 1, so the caller has to have a successor
        phase for every frame it passes. On the last step of a horizon there is
        none, and the right treatment is to leave that frame out rather than to
        hold the reference past its end -- the held value would be a fabricated
        continuation of the demonstration.

        `actuation` is the (N, k) gripper accumulator column of the frames, from
        `TrackingResult.actuation`. See `flow`.
        """
        assert_action_side_caliber(caliber)
        states, steps, actuation = self._frames(states, steps, actuation)
        method = DerivativeMethod(method)
        snapshot = self._snapshot()
        try:
            if method is DerivativeMethod.FLOW:
                out = self._a_dot_flow(states, steps, actuation)
            else:
                out = self._a_dot_jacobian(states, steps, actuation)
        finally:
            self._restore(snapshot)
        return out

    def _a_dot_flow(
        self, states: np.ndarray, steps: np.ndarray, actuation: np.ndarray
    ) -> np.ndarray:
        rows = []
        for state, t, act in zip(states, steps, actuation, strict=True):
            a0 = np.asarray(self.policy(state, int(t)), dtype=np.float64)
            nxt = self.flow(state, a0, actuation=act)
            a1 = np.asarray(self.policy(nxt, int(t) + 1), dtype=np.float64)
            rows.append((a1 - a0) / self.dt)
        return np.asarray(rows, dtype=np.float64)

    def _a_dot_jacobian(
        self, states: np.ndarray, steps: np.ndarray, actuation: np.ndarray
    ) -> np.ndarray:
        h = self.fd_step
        rows = []
        for state, t, act in zip(states, steps, actuation, strict=True):
            step = int(t)
            a0 = np.asarray(self.policy(state, step), dtype=np.float64)
            x_dot = (self.flow(state, a0, actuation=act) - state) / self.dt
            jac = np.empty((a0.size, state.size), dtype=np.float64)
            for k in range(state.size):
                plus = state.copy()
                minus = state.copy()
                plus[k] += h
                minus[k] -= h
                jac[:, k] = (
                    np.asarray(self.policy(plus, step), dtype=np.float64)
                    - np.asarray(self.policy(minus, step), dtype=np.float64)
                ) / (2.0 * h)
            explicit = (
                np.asarray(self.policy(state, step + 1), dtype=np.float64) - a0
            ) / self.dt
            rows.append(jac @ x_dot + explicit)
        return np.asarray(rows, dtype=np.float64)

    def split(
        self,
        states: np.ndarray,
        steps: np.ndarray,
        *,
        caliber: DelayCaliber,
        actuation: np.ndarray,
    ) -> DerivativeSplit:
        """The convective and explicit halves of `a_dot`, measured separately.

        Three policy evaluations per frame instead of the flow method's two:
        the extra one is the teacher at the same state and the next phase, which
        is what isolates the term a stationary policy does not have.
        """
        assert_action_side_caliber(caliber)
        states, steps, actuation = self._frames(states, steps, actuation)
        snapshot = self._snapshot()
        try:
            convective, explicit, total = [], [], []
            for state, t, act in zip(states, steps, actuation, strict=True):
                step = int(t)
                a0 = np.asarray(self.policy(state, step), dtype=np.float64)
                nxt = self.flow(state, a0, actuation=act)
                convective.append(
                    (np.asarray(self.policy(nxt, step), dtype=np.float64) - a0) / self.dt
                )
                explicit.append(
                    (np.asarray(self.policy(state, step + 1), dtype=np.float64) - a0)
                    / self.dt
                )
                total.append(
                    (np.asarray(self.policy(nxt, step + 1), dtype=np.float64) - a0)
                    / self.dt
                )
        finally:
            self._restore(snapshot)
        return DerivativeSplit(
            convective=np.asarray(convective, dtype=np.float64),
            explicit=np.asarray(explicit, dtype=np.float64),
            total=np.asarray(total, dtype=np.float64),
        )

    def cross_check(
        self,
        states: np.ndarray,
        steps: np.ndarray,
        *,
        caliber: DelayCaliber,
        actuation: np.ndarray,
    ) -> DerivativeReport:
        """Report the gap between the flow and the explicit-Jacobian calibers."""
        flow = self.a_dot(
            states, steps, caliber=caliber, actuation=actuation,
            method=DerivativeMethod.FLOW,
        )
        jacobian = self.a_dot(
            states, steps, caliber=caliber, actuation=actuation,
            method=DerivativeMethod.JACOBIAN,
        )
        diff = np.abs(flow - jacobian)
        scale = float(np.linalg.norm(flow))
        return DerivativeReport(
            max_abs_difference=float(diff.max()) if diff.size else 0.0,
            relative_difference=float(np.linalg.norm(diff) / scale) if scale > 0.0 else 0.0,
            n_states=int(np.atleast_2d(states).shape[0]),
        )
