"""Closed-form compensation fields psi_j, and the evaluator for c_delta.

Definition 3.1 (5.1) fixes what these compute:

    c_delta(x, z) = pi_{e,delta,tau}(x, z) - pi_{e,0,tau}(x)

evaluated at the *clean reference policy's* action, not at the policy under
test. 9.2.10 makes analytic writability the primary screening criterion for the
whole catalogue, because Lambda's numerator and denominator both need a per-frame
direction field and because a surrogate can only be matched against a field that
is known in closed form. `params.analytic_field` says which entries qualify;
D7 and D9 do not, and their classes raise rather than return an approximation --
a numerical twin difference is a different object, and mixing the two would put
un-matchable fields into the Lambda^exc null distribution.

Units. Every field returns action units, which is what the diagnostics consume.
9.6 fixes one action unit at 50 mm of translation and 0.5 rad of rotation, and
the conversion is read from the controller's `output_max` rather than written as
a literal. The gripper column is zero throughout: none of the ten defects acts on
it.

Two mappings do all the work.

*Task-space offset to action*: `dx / output_max`. D1 and D10 land here directly,
since an encoder offset displaces the perceived end-effector by `J dq` and the
teacher has to command the opposite.

*Joint torque to action*: an unmodelled joint torque tau is balanced in steady
state by an OSC wrench W with `J^T W = tau`, and OSC realises a wrench of
`kp dx`, so the teacher has to shift its commanded pose by `pinv(J^T) tau / kp`.
That is the concrete content of 5.3's `c = K^-1 tau_g` with
`K = J^T diag(kp) J`, written in the task space where OSC's gains live rather
than in a joint space where K is rank deficient on a 7 DoF arm.

Memory-state and action-dependent inputs arrive through the constructor, never as
a defaulted argument: D4's engagement state z, D5's and D8's clean-policy torque,
D2's `a_dot` and D10's session clock are all things the caller has to produce,
and substituting a plausible one would be a fabricated ground truth
(CONTRACT 3.6).

The second-order hook. 9.2.2 states D2's closed form as
`c = ell a_dot + O(ell^2)`, and the same question -- how much of the field is
first order in the dose -- applies to every entry. `verify_second_order`
evaluates the field at dose eps and 2 eps and reports `c(2 eps) - 2 c(eps)`,
which is `2 eps^2 c'' + O(eps^3)`; a field that is exactly linear in its
parameter returns zero and one that is not returns its curvature share. D2 also
gets the physical term the formula drops, `0.5 ell^2 a_ddot`, through
`D2Field.second_order_term`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from portal_trace.defects.adot import delay_seconds
from portal_trace.defects.delta import Delta

__all__ = [
    "NoAnalyticField",
    "FieldContext",
    "CompensationField",
    "SecondOrderReport",
    "AnalyticField",
    "D1Field",
    "D2Field",
    "D3aField",
    "D3bField",
    "D4Field",
    "D5Field",
    "D6Field",
    "D7Field",
    "D8Field",
    "D9Field",
    "D10Field",
    "field_for",
    "apply_pinv",
    "apply_pinv_transpose",
]


class NoAnalyticField(NotImplementedError):
    """This defect has a numerical ground truth only (9.2.7, 9.2.10).

    Raised rather than approximated. A field that is not analytically writable
    cannot enter the surrogate matching -- 9.2.11 matches four dimensions of a
    *known* direction field -- so returning a twin-difference estimate here would
    put an object of a different kind into Lambda^exc.
    """


@dataclass(frozen=True)
class FieldContext:
    """Everything a field needs about the family, resolved by name.

    Built from a `DefectWrapper` so that the indices and gains a field uses are
    the same ones the injector used, rather than a second transcription of them.
    """

    qpos_index: tuple[int, ...]
    qvel_index: tuple[int, ...]
    site_id: int
    kp: np.ndarray            # (6,) OSC task-space gains
    output_max: np.ndarray    # (6,) one action unit, in m and rad
    action_dim: int

    @classmethod
    def from_wrapper(cls, wrapper: Any) -> FieldContext:
        controller = wrapper.robot.part_controllers[wrapper.arm]
        index = wrapper.index
        return cls(
            qpos_index=index.qpos_index,
            qvel_index=index.qvel_index,
            site_id=index.ref_site_id,
            kp=np.asarray(controller.kp, dtype=np.float64)[:6].copy(),
            output_max=np.asarray(controller.output_max, dtype=np.float64)[:6].copy(),
            action_dim=int(wrapper.env.action_dim),
        )

    @property
    def n_joints(self) -> int:
        return len(self.qvel_index)


@runtime_checkable
class CompensationField(Protocol):
    """The interface the diagnostics consume."""

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        """(N, nq + nv) states in, (N, action_dim) compensation out."""


@dataclass(frozen=True)
class SecondOrderReport:
    """How much of a closed form is not first order in the dose."""

    probe: float
    first_order_norm: float
    residual_norm: float

    @property
    def residual_share(self) -> float:
        if self.first_order_norm == 0.0:
            return 0.0
        return self.residual_norm / self.first_order_norm


# Above this condition number of J J^T the Gram route below is not used. The
# Gram squares the condition number of J, so it loses about half the significant
# digits, and the loss is linear in cond(J J^T): the relative deviation from
# `np.linalg.pinv` measures 1.0e-10 at 3.1e3, 1.7e-9 at 2.8e5 and 5.6e-5 at
# 3.0e9. This limit keeps the fast route inside 1e-10, which is the accuracy the
# per-frame ground truth is compared at everywhere else, and hands the rest to
# the SVD. It is a cutover between two exact routes, not a degraded mode.
GRAM_CONDITION_LIMIT = 1.0e4


def _gram_pinv_apply(jac: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(J J^T)^+ v per frame, and the frames the Gram route is too blunt for.

    Both pseudo-inverses this module and its callers need factor through the
    6x6 Gram matrix, by the identity A^+ = A^T (A A^T)^+:

        pinv(J^T) tau = (J J^T)^+ (J tau)      joint torque -> task wrench
        pinv(J)  dx   = J^T (J J^T)^+ dx       task offset  -> joint offset

    so the expensive part is shared and is done once here. `np.linalg.pinv` on
    the stack costs one SVD per frame, and these run over roughly 45 M frames
    for the primary endpoint. A symmetric eigendecomposition of the Gram matrix
    is about twice as fast: writing J = U S V^T makes J J^T = U S^2 U^T, so the
    eigenvectors are U and the eigenvalues are the squared singular values.

    Rank deficiency inside the fast route is handled the way `pinv` handles it
    rather than avoided -- the same relative cutoff zeroes the directions the
    arm cannot resist. Plain `solve` on the Gram matrix would be faster still
    and would return nonsense exactly at the kinematic singularities where these
    fields are largest.
    """
    eigenvalues, basis = np.linalg.eigh(jac @ np.swapaxes(jac, 1, 2))
    cutoff = (max(jac.shape[-1], jac.shape[-2]) * np.finfo(np.float64).eps) * eigenvalues[:, -1:]
    keep = eigenvalues > cutoff
    inverse = np.where(keep, 1.0 / np.where(keep, eigenvalues, 1.0), 0.0)
    projected = np.swapaxes(basis, 1, 2) @ v[:, :, None]
    out = (basis @ (inverse[:, :, None] * projected))[:, :, 0]
    return out, eigenvalues[:, 0] * GRAM_CONDITION_LIMIT < eigenvalues[:, -1]


def apply_pinv_transpose(jac: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """pinv(J^T) tau: the task-space wrench that balances a joint torque.

    Public because `teacher.project` builds the same D3 column of the physical
    basis psi and must build it out of the same map as the ground-truth field,
    not out of a second transcription of it.
    """
    out, ill = _gram_pinv_apply(jac, (jac @ tau[:, :, None])[:, :, 0])
    if ill.any():
        out[ill] = np.einsum(
            "nij,nj->ni", np.linalg.pinv(np.swapaxes(jac[ill], 1, 2)), tau[ill]
        )
    return out


def apply_pinv(jac: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """pinv(J) dx: the minimum-norm joint offset realising a task-space offset."""
    gram_solved, ill = _gram_pinv_apply(jac, dx)
    out = (np.swapaxes(jac, 1, 2) @ gram_solved[:, :, None])[:, :, 0]
    if ill.any():
        out[ill] = np.einsum("nij,nj->ni", np.linalg.pinv(jac[ill]), dx[ill])
    return out


def _mj_batch(
    states: np.ndarray, model: Any, data: Any, ctx: FieldContext, *, gravity: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Per-frame Jacobians, joint states and optionally tau_g, in one pass.

    The MuJoCo calls are inherently per frame -- J(q) is not a batched object --
    so this is the one loop in the module. Everything downstream of it is
    vectorised over N, and the diagnostics call it on `FrameTable` chunks rather
    than on the whole 45 M-frame table.

    The Jacobian and the gravity torque share `mj_kinematics` and `mj_comPos`,
    which together are most of the per-frame cost, so `gravity=True` gets tau_g
    out of the same pass rather than walking the batch twice. It costs nothing
    in fidelity: `mj_jacSite` is a position-stage quantity and does not read
    qvel, and RNEA is run at qvel = 0 exactly as 9.2.3 requires -- the loop
    writes zeros into `data.qvel` and the caller's qvel is sliced out of the
    state array, never read back from `mjData`.
    """
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    nq, nv = int(model.nq), int(model.nv)
    if states.shape[1] != nq + nv:
        raise ValueError(f"states are {states.shape[1]} wide, this model needs {nq + nv}")
    n = states.shape[0]
    qpos_idx = np.asarray(ctx.qpos_index, dtype=np.intp)
    qvel_idx = np.asarray(ctx.qvel_index, dtype=np.intp)

    jac = np.empty((n, 6, qvel_idx.size), dtype=np.float64)
    tau_g = np.empty((n, qvel_idx.size), dtype=np.float64) if gravity else None
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    scratch = np.zeros(nv)
    data.qvel[:] = 0.0

    for i, state in enumerate(states):
        data.qpos[:] = state[:nq]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, jacr, ctx.site_id)
        jac[i, :3, :] = jacp[:, qvel_idx]
        jac[i, 3:, :] = jacr[:, qvel_idx]
        if tau_g is not None:
            mujoco.mj_comVel(model, data)
            mujoco.mj_rne(model, data, 0, scratch)
            tau_g[i] = scratch[qvel_idx]
    return jac, states[:, qpos_idx], states[:, nq:][:, qvel_idx], tau_g


def _forward_batch(
    states: np.ndarray, model: Any, data: Any, ctx: FieldContext
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame Jacobians and joint states."""
    jac, qpos, qvel, _ = _mj_batch(states, model, data, ctx, gravity=False)
    return jac, qpos, qvel


@dataclass(frozen=True)
class AnalyticField:
    """Common machinery: the two unit mappings and the second-order hook."""

    delta: Delta
    ctx: FieldContext

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        raise NotImplementedError

    # -- unit mappings ---------------------------------------------------

    def _pad(self, task: np.ndarray) -> np.ndarray:
        """(N, 6) task-space offset in m/rad -> (N, action_dim) action units."""
        out = np.zeros((task.shape[0], self.ctx.action_dim), dtype=np.float64)
        out[:, :6] = task / self.ctx.output_max
        return out

    def _from_task_offset(self, dx: np.ndarray) -> np.ndarray:
        return self._pad(dx)

    def _from_joint_torque(self, tau: np.ndarray, jac: np.ndarray) -> np.ndarray:
        """pinv(J^T) tau / kp, batched over N."""
        return self._pad(apply_pinv_transpose(jac, tau) / self.ctx.kp)

    # -- second-order hook ------------------------------------------------

    def verify_second_order(
        self, states: np.ndarray, model: Any, data: Any, *, probe: float = 0.5
    ) -> SecondOrderReport:
        """Curvature of the closed form in its own dose parameter.

        `c(2 eps) - 2 c(eps) = 2 eps^2 c'' + O(eps^3)`. A field that is exactly
        first order returns zero; the D2 term 9.2.2 actually drops is physical
        rather than parametric and lives in `D2Field.second_order_term`.
        """
        if probe <= 0.0:
            raise ValueError(f"probe dose {probe} is not positive")
        single = replace(self, delta=self.delta.scaled(probe))
        double = replace(self, delta=self.delta.scaled(2.0 * probe))
        c1 = single.evaluate(states, model, data)
        c2 = double.evaluate(states, model, data)
        return SecondOrderReport(
            probe=float(probe),
            first_order_norm=float(np.linalg.norm(2.0 * c1)),
            residual_norm=float(np.linalg.norm(c2 - 2.0 * c1)),
        )


# ---------------------------------------------------------------------------
# The analytically writable fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D1Field(AnalyticField):
    """c = J(q) dq, constant in the joint frame (9.2.1, 5.3).

    The task-space image varies only through J, which is close to constant in the
    neighbourhood of the terminal fine-approach phase -- which is also why 14.5
    E-11 rules out a state permutation as a null distribution: permuting a
    near-constant field returns the field.
    """

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _ = _forward_batch(states, model, data, self.ctx)
        dq = self.delta.joint_values("delta_q")
        return self._from_task_offset(np.einsum("nij,j->ni", jac, dq))


@dataclass(frozen=True)
class D2Field(AnalyticField):
    """c = ell * a_dot (9.2.2, 5.1).

    `a_dot` is supplied per frame by `adot.ClosedLoopDerivative`, which is where
    the action-side caliber is asserted; this class only scales it. It is not
    recomputed here, because the derivative needs the clean twin and the teacher
    policy and this module owns neither.
    """

    a_dot: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.a_dot is None:
            raise ValueError(
                "D2's ground truth is c = ell * a_dot, and a_dot has to come from "
                "adot.ClosedLoopDerivative on the clean twin (5.1). There is no "
                "state-only expression for it, so it is a required input."
            )

    @property
    def ell(self) -> float:
        return delay_seconds(self.delta.params["n_control_steps"])

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        n = np.atleast_2d(np.asarray(states)).shape[0]
        a_dot = np.atleast_2d(np.asarray(self.a_dot, dtype=np.float64))
        if a_dot.shape[0] != n:
            raise ValueError(f"a_dot has {a_dot.shape[0]} frames, states have {n}")
        return self.ell * a_dot

    def second_order_term(self, a_ddot: np.ndarray) -> np.ndarray:
        """The O(ell^2) term the closed form drops: 0.5 ell^2 a_ddot.

        9.2.2 states the first-order form as correct and the remainder as
        O(ell^2); this makes the remainder available so that "correct" can be
        reported as a number at each of the four levels rather than asserted.
        """
        return 0.5 * self.ell**2 * np.atleast_2d(np.asarray(a_ddot, dtype=np.float64))


@dataclass(frozen=True)
class _GravityDroopField(AnalyticField):
    """Shared body of D3a and D3b: c = K^-1 tau_g(q), configuration dependent.

    Configuration dependence is the point (9.2.3): a per-family fixed affine
    action normalisation is a constant map and cannot absorb a field that moves
    with q, which is what makes D3 immune to confounder X7 and D8 not.
    """

    def _residual_torque(self, tau_g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _, tau_g = _mj_batch(states, model, data, self.ctx, gravity=True)
        assert tau_g is not None
        return self._from_joint_torque(self._residual_torque(tau_g), jac)


@dataclass(frozen=True)
class D3bField(_GravityDroopField):
    """The purely static caliber: the uncompensated fraction is sigma_g tau_g."""

    def _residual_torque(self, tau_g: np.ndarray) -> np.ndarray:
        return self.delta.joint_values("sigma_g") * tau_g


@dataclass(frozen=True)
class D3aField(AnalyticField):
    """c = J(q) tau_g(q) / k, the deflection of a series-elastic transmission.

    9.2.3 caliber (a) puts a co-located second joint carrying `stiffness=k`
    between the motor and the load. The actuator delivers the torque it was
    asked for; what the gravity load does is deflect that spring, by
    `dq = tau_g / k` radians, and the tool sags by `J dq`. So this is a
    *kinematic* offset and takes the same route as D1's encoder bias -- not
    D3b's route. D3b is a torque deficit: its caliber scales the controller's
    own gravity term, so the missing `sigma_g tau_g` has to be supplied through
    the task-space wrench, and `pinv(J^T) tau / kp` is what supplying it costs.
    Sending D3a down that route would price a spring deflection as if the
    controller had to generate it, and would need a joint-space stiffness that
    the OSC gain vector -- which is in N/m and rad-scaled N*m/rad on task axes,
    not per joint -- cannot supply.

    9.2.3 keeps this caliber out of the 36 pairs because it also injects a
    4.5 Hz mode inside the control bandwidth; the field is still analytic, and
    E1 and E3 use it.
    """

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _, tau_g = _mj_batch(states, model, data, self.ctx, gravity=True)
        assert tau_g is not None
        k = self.delta.joint_values("k_transmission")
        if np.any(k <= 0.0):
            raise ValueError(f"transmission stiffness must be positive, got {k}")
        return self._from_task_offset(np.einsum("nij,nj->ni", jac, tau_g / k))


@dataclass(frozen=True)
class D4Field(AnalyticField):
    """The inverse play operator, (b_j / 2) sign(qdot_j) e_j, on the re-engagement leg.

    Non-zero only while the joint is traversing the slack after a reversal, which
    is a function of the memory state z rather than of x -- (A5)(ii) is what makes
    the pointwise theory apply, and 5.1 says so explicitly. `engaged` is that
    state, read from the high-fidelity caliber's `efc_pos` (the equality
    constraint is always active, which is why 9.2.4 chose it) or from the play
    operator's `s`. There is no default: guessing it would invent the support of
    the field.
    """

    engaged: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.engaged is None:
            raise ValueError(
                "D4's compensation is supported only on the re-engagement leg, and "
                "that is decided by the engagement state z, not by x. Pass the "
                "per-frame boolean array extracted from DefectWrapper.defect_state() "
                "with defects.inject.engagement_flags; D4hi snapshots derive that "
                "slice from the shadow equality's efc_pos (9.2.4, 5.1)."
            )

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, qvel = _forward_batch(states, model, data, self.ctx)
        engaged = np.atleast_2d(np.asarray(self.engaged))
        if engaged.shape != qvel.shape:
            raise ValueError(
                f"engagement state is {engaged.shape}, expected {qvel.shape} "
                "(one flag per joint per frame)"
            )
        half_band = 0.5 * self.delta.joint_values("backlash")
        dq = half_band * np.sign(qvel) * (~engaged.astype(bool)).astype(np.float64)
        return self._from_task_offset(np.einsum("nij,nj->ni", jac, dq))


@dataclass(frozen=True)
class D5Field(AnalyticField):
    """c = sign(u) u_dz (9.2.5, 5.3).

    The dead zone eats `w` of torque magnitude, so the teacher restores it by
    commanding `w sign(u)` more. `u` is the clean reference policy's joint torque
    at that frame: 5.1 pins the evaluation point to `pi_{e,0,tau}`, not to the
    policy under test, and the version evaluated at the tested policy is the
    sensitivity analysis that section asks for separately.
    """

    torque: np.ndarray = None  # type: ignore[assignment]
    torque_limit: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.torque is None or self.torque_limit is None:
            raise ValueError(
                "D5's field is sign(u) w, so it needs the clean policy's joint "
                "torque u per frame and the per-joint rated torque that fixes w. "
                "Both come from the clean twin (5.1); neither has a default."
            )

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _ = _forward_batch(states, model, data, self.ctx)
        u = np.atleast_2d(np.asarray(self.torque, dtype=np.float64))
        width = self.delta.joint_values("w_over_tau_max") * np.asarray(
            self.torque_limit, dtype=np.float64
        )
        return self._from_joint_torque(np.sign(u) * width, jac)


@dataclass(frozen=True)
class D6Field(AnalyticField):
    """c = 0.5 (f+ - f-) sign(qdot) (9.2.6, 5.3).

    Only the asymmetric half appears: the symmetric base is direction independent
    and the teacher's feedforward for it cancels between the two travel
    directions.
    """

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, qvel = _forward_batch(states, model, data, self.ctx)
        half = 0.5 * self.delta.joint_values("delta_f")
        return self._from_joint_torque(half * np.sign(qvel), jac)


@dataclass(frozen=True)
class D8Field(AnalyticField):
    """c = (1/g - 1) u, the only defect class for which gain asymmetry holds (A-12).

    Multiplicative and configuration independent, which is exactly the form a
    global affine action normalisation absorbs -- 9.2.8 requires its Lambda to be
    reported next to D3's for that reason, the difference being the size of
    confounder X7.
    """

    torque: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.torque is None:
            raise ValueError(
                "D8 scales the commanded torque, so its field needs the clean "
                "policy's joint torque per frame (5.1)"
            )

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _ = _forward_batch(states, model, data, self.ctx)
        g = self.delta.joint_values("gain")
        if np.any(g <= 0.0):
            raise ValueError(f"a transmission gain of {g} is not invertible")
        u = np.atleast_2d(np.asarray(self.torque, dtype=np.float64))
        return self._from_joint_torque(u * (1.0 / g - 1.0), jac)


@dataclass(frozen=True)
class D10Field(AnalyticField):
    """D1's field with 9.2.9's drift law: dq(t) = dq_inf (1 - exp(-t/T)).

    `session_time` is per frame and runs on session time, not episode time. If it
    restarted each episode the field would be a fixed function repeated
    identically on every trajectory, which is a constant defect with an internal
    time axis and does not break (A4) -- and breaking (A4) is D10's entire
    purpose (9.2.9, 14.5 E-5).
    """

    session_time: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.session_time is None:
            raise ValueError(
                "D10's offset depends on elapsed session time, which is wrapper "
                "state rather than a function of x. Pass the per-frame clock from "
                "defect_state() (9.2.0 discipline 3)."
            )

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        jac, _, _ = _forward_batch(states, model, data, self.ctx)
        t = np.asarray(self.session_time, dtype=np.float64).reshape(-1, 1)
        if t.shape[0] != jac.shape[0]:
            raise ValueError(f"session_time has {t.shape[0]} frames, states have {jac.shape[0]}")
        asymptote = self.delta.joint_values("delta_q_inf")
        if "tau_thermal" in self.delta.params:
            progress = 1.0 - np.exp(-t / float(self.delta.params["tau_thermal"]))
        else:
            progress = 1.0 - np.exp(-t / float(self.delta.params["s0_thermal"]))
        dq = asymptote[None, :] * progress
        return self._from_task_offset(np.einsum("nij,nj->ni", jac, dq))


@dataclass(frozen=True)
class D7Field(AnalyticField):
    """9.2.7: numerical ground truth only."""

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        raise NoAnalyticField(
            "D7 (structural compliance vibration) has no closed-form compensation "
            "direction (9.2.7, 9.2.10). Its ground truth is the twin difference, "
            "which cannot enter the surrogate matching of 9.2.11 and therefore "
            "cannot enter Lambda^exc. D7 is kept as the zero-GPU open-loop replay "
            "carrier that appendix A-21 forbids dropping, not as a Lambda cell."
        )


@dataclass(frozen=True)
class D9Field(AnalyticField):
    """9.2.7: numerical ground truth only."""

    def evaluate(self, states: np.ndarray, model: Any, data: Any) -> np.ndarray:
        raise NoAnalyticField(
            "D9 (cogging torque) has no closed-form compensation direction "
            "(9.2.7, 9.2.10). The injected torque -A sin(N q + phi) is analytic; "
            "the feedforward a teacher would learn against it is not, and it is "
            "the latter that Lambda needs."
        )


_FIELD_BY_CODE: dict[str, type[AnalyticField]] = {
    "D1": D1Field,
    "D2": D2Field,
    "D3a": D3aField,
    "D3b": D3bField,
    "D4hi": D4Field,
    "D4lo": D4Field,
    "D5": D5Field,
    "D6": D6Field,
    "D7": D7Field,
    "D8": D8Field,
    "D9": D9Field,
    "D10": D10Field,
}


def field_for(delta: Delta, ctx: FieldContext, **extra: Any) -> AnalyticField:
    """The field class for this delta, with its required extra inputs bound."""
    cls = _FIELD_BY_CODE[delta.defect_code]
    return cls(delta=delta, ctx=ctx, **extra)


def _gravity_torque_batch(
    states: np.ndarray, model: Any, data: Any, ctx: FieldContext
) -> np.ndarray:
    """tau_g(q) per frame, from RNEA at qdot = 0.

    Not `qfrc_bias`: 9.2.3 verified that it is gravity plus Coriolis plus
    centrifugal, and taking it here would make a supposedly static field
    velocity dependent.
    """
    _, _, _, tau_g = _mj_batch(states, model, data, ctx, gravity=True)
    assert tau_g is not None
    return tau_g
