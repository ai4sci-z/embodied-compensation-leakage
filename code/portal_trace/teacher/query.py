"""The off-branch query of the teacher: per-frame ground truth for c_delta (9.6.7).

Definition 3.1 fixes the compensation field as a difference of two policies
evaluated at the same state,

    c_delta(x, z) = pi_{e,delta,tau}(x, z) - pi_{e,0,tau}(x),

and 9.6.7 pins the implementation down to a formula that can actually be run,
because the teacher's ILC feedforward is indexed by control step rather than by
state:

    c_true[t] = clip(OSC_delta(x_t; x_ref[t]) + u_inf_delta[t])
              - clip(OSC_0(x_t; x_ref[t]) + u_inf_0[t]).

Three things in that line are easy to get wrong and are therefore structural
here rather than left to the caller.

**The clip is per side.** 9.6.7 item 2: the action box is [-1, 1] and clipping
is not linear, so clipping the difference is a different quantity from
differencing the clips. The main caliber is the second one -- it is the
difference between actions that exist in the data -- and `CTrue` carries the
saturation flag alongside, because 3.4.2 requires the saturated-frame share to
be reported per cell and annotated above 5%.

**Both sides carry an ILC feedforward.** 9.6.7 item 3: u_inf on the clean twin
is not zero. It compensates the tracker's own phase lag, which has nothing to do
with the defect, and dropping it would book that lag as compensation. Neither
`TeacherArm` has a default for it.

**One state, two models.** 9.6.7 item 1: a twin pair compiles to the same nq and
nv, so the recorded state is written into both models unchanged. For the defects
that alter geometry -- D6's link scale, D3a's series elasticity -- the two models
disagree about where the tool is at that same qpos, and that disagreement is part
of the answer.

**Two models is not always two mjModels.** D1 and D10 write nothing into the
model at all: their defect is that the controller *perceives* a shifted qpos,
and 9.2.1's trap is that biasing `controller.joint_pos` alone cancels itself
because OSC closes its loop on `site_xpos` from the true kinematics. The live
wrapper runs a sensing twin for exactly that reason, and so does this module --
without it the two arms of a D1 pair would agree on the tool pose and c_true
would report only the difference of the feedforwards. D10's bias moves with the
frame's session clock, which is why the memory state z is read per frame.

**Cost.** The primary endpoint runs this over roughly 45 M frames, so the module
is written batched: `states` and `t` come in as arrays and one array comes back.
The only per-frame work left is forward kinematics, which MuJoCo does not batch
(the same loop `defects.fields._forward_batch` documents); everything downstream
of the tool pose is vectorised over N.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from portal_trace.defects import adot, inject
from portal_trace.defects.delta import Delta
from portal_trace.defects.inject import TwinPair
from portal_trace.defects.params import WritableField
from portal_trace.registry.tasks import HORIZON
from portal_trace.teacher.osc import OSCTracker
from portal_trace.teacher.refs import ReferenceTrajectory

__all__ = [
    "TeacherArm",
    "TwinQuery",
    "CTrue",
    "SATURATION_ANNOTATION_SHARE",
    "MEMORY_DEFECTS",
    "TASK_CHANNELS",
    "query_clean",
    "a_dot_nominal",
    "a_dot_split",
    "arm_from_wrapper",
    "twin_query",
]

# The six task-space channels of the 9.6.1 action. a_dot is read on these alone:
# the gripper column is a discrete open/close command, 9.4.4's psi block for D2
# is six wide, and the ILC feedforward that 9.4.7 compares it against never had
# a seventh column to begin with.
TASK_CHANNELS = 6

# 9.6.7 item 2 / 10.1.4: above this share of saturated frames the cell's Lambda
# carries an annotation, because clipping flattens the large values of c_delta
# and biases the denominator down. Stated numerically in the proposal, so a
# constant rather than a lock-file measurement.
SATURATION_ANNOTATION_SHARE = 0.05

# 5.1: the defects whose compensation depends on the wrapper's memory state z.
# D2 is the action ring buffer, D4 the engagement flags, D10 the drift
# accumulator. For these the off-branch query is `query_clean(x, t, z)` and z has
# to be restored before evaluation, or the query is not reproducible (9.2.0
# discipline 3).
MEMORY_DEFECTS = frozenset({"D2", "D4", "D10"})


def _site_poses(
    model: Any,
    data: Any,
    states: np.ndarray,
    site_id: int,
    *,
    offsets: np.ndarray | None = None,
    qpos_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, nq+nv) states -> (N, 3) tool positions and (N, 3, 3) tool orientations.

    `mj_kinematics` is the whole cost: the tool pose is a function of qpos alone,
    so qvel is not written and neither `mj_comPos` nor `mj_crb` is called. This
    is the one loop in the module and it is over MuJoCo, which has no batched
    forward-kinematics entry point.

    `offsets` is the (N, n_joints) perceived-encoder bias of 9.2.1, added to the
    arm's qpos before the kinematics run. It is what makes this the *perceived*
    tool pose rather than the true one; see `TeacherArm._sensing_offsets`.
    """
    nq = int(model.nq)
    n = states.shape[0]
    qpos = np.array(states[:, :nq], dtype=np.float64)
    if offsets is not None:
        qpos[:, qpos_index] += offsets
    pos = np.empty((n, 3), dtype=np.float64)
    rot = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        pos[i] = data.site_xpos[site_id]
        rot[i] = data.site_xmat[site_id].reshape(3, 3)
    return pos, rot


@dataclass
class TeacherArm:
    """One side of the 9.6.7 difference, evaluable at arbitrary (x, t).

    Holds a compiled body, the reference the demonstration tracked, and the ILC
    feedforward that converged on that body. The three travel together because
    the teacher operator O(e, delta, tau) is exactly that triple: a different
    u_inf is a different teacher, and pairing one body's states with another
    body's feedforward is the silent error this dataclass exists to prevent.

    Queries do not touch the live simulator. Forward kinematics runs on a private
    `MjData`, so an off-branch query in the middle of a rollout leaves the episode
    where it was -- 9.6.7 evaluates counterfactual states that the body never
    visits, and doing that in place would corrupt the rollout being annotated.
    """

    wrapper: Any                      # DefectWrapper
    reference: ReferenceTrajectory
    u_inf: np.ndarray                 # (T, 6) converged ILC feedforward, action units
    tracker: OSCTracker
    site_id: int
    base_pos: np.ndarray
    base_rot: np.ndarray
    _model: Any
    _data: Any
    _qpos_index: np.ndarray          # arm joints within qpos, for the sensing twin

    def __post_init__(self) -> None:
        if self.u_inf.shape != (self.reference.horizon, 6):
            raise ValueError(
                f"u_inf is {self.u_inf.shape}, expected "
                f"({self.reference.horizon}, 6). 9.6.7 item 3: both sides of the "
                "difference carry their own converged feedforward and neither may "
                "be stood in for by zero"
            )

    @property
    def delta(self) -> Delta:
        return self.wrapper.delta

    @property
    def needs_memory_state(self) -> bool:
        return self.delta.defect_code in MEMORY_DEFECTS and not self.delta.is_zero

    def raw_actions(
        self, states: np.ndarray, t: np.ndarray, z: np.ndarray | None = None
    ) -> np.ndarray:
        """(N, nq+nv), (N,) -> (N, ACTION_DIM) teacher actions before the clip.

        The unclipped values are what `CTrue` needs to report saturation, so they
        are produced here and the clip is applied by the caller that owns the
        difference.
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.float64))
        t = np.atleast_1d(np.asarray(t, dtype=np.int64))
        if t.shape[0] == 1 and states.shape[0] != 1:
            t = np.broadcast_to(t, (states.shape[0],))
        if t.shape != (states.shape[0],):
            raise ValueError(f"t is {t.shape}, states have {states.shape[0]} frames")
        if t.min() < 0 or t.max() >= self.reference.horizon:
            raise ValueError(
                f"control-step indices span [{t.min()}, {t.max()}], the reference "
                f"has {self.reference.horizon} steps"
            )
        nq, nv = int(self._model.nq), int(self._model.nv)
        if states.shape[1] != nq + nv:
            raise ValueError(
                f"states are {states.shape[1]} wide, this body needs nq + nv = "
                f"{nq + nv}. 9.6.7 item 1 moves a state between the twins unchanged, "
                "so a width mismatch is a pairing error, not something to reshape"
            )

        offsets = self._sensing_offsets(states.shape[0], z)
        pos, rot = _site_poses(
            self._model,
            self._data,
            states,
            self.site_id,
            offsets=offsets,
            qpos_index=self._qpos_index,
        )

        delta = self.tracker.delta_action(
            self.reference.positions[t],
            self.reference.rotations[t],
            pos,
            rot,
            self.base_pos,
            self.base_rot,
        )
        return self.tracker.compose_action(delta, self.reference.gripper[t], self.u_inf[t])

    def actions(
        self, states: np.ndarray, t: np.ndarray, z: np.ndarray | None = None
    ) -> np.ndarray:
        """The same thing clipped to the action box, which is what the data holds."""
        return self.tracker.clip_action(self.raw_actions(states, t, z))

    def as_policy(self) -> Callable[[np.ndarray, int], np.ndarray]:
        """This arm as the `adot.Policy` protocol: one (state, control step) at a time.

        `defects.adot.ClosedLoopDerivative` differences the teacher along the
        clean twin's flow and needs it callable one query at a time; the batched
        `actions` is the entry point everything else uses, so the adapter is a
        thin wrapper over it rather than a second evaluation path.

        The phase travels with the state instead of being bound once, because
        5.1's a_dot is the derivative of the emitted action *sequence* and this
        teacher's action moves with the control step through both x_ref[t] and
        u_inf[t]. Freezing t here would silently produce grad_x a . f alone.
        """

        def policy(state: np.ndarray, t: int) -> np.ndarray:
            return self.actions(np.asarray(state, dtype=np.float64)[None, :], int(t))[0]

        return policy

    def _sensing_offsets(self, n: int, z: np.ndarray | None) -> np.ndarray | None:
        """The (N, n_joints) encoder bias this body's controller perceives, or None.

        9.2.1's trap, restated for the off-branch query: a joint offset that is
        added to `controller.joint_pos` alone cancels itself, because OSC closes
        the loop on `site_xpos` and that comes out of the true forward
        kinematics. The live wrapper avoids it by recomputing pose, orientation
        and Jacobian on a sensing twin (`DefectWrapper._write_sensing_twin`);
        this method is that same twin for a query, and without it the D1 and D10
        arms of a pair would return the identical tool pose and c_true would
        collapse onto the difference of the two feedforwards.

        For D10 the bias is a function of the frame's own session clock, so it is
        read out of `z` per frame rather than off the wrapper. That also keeps
        the query side-effect free: 9.6.7 evaluates counterfactual states, and a
        query that wrote the accumulator would advance the very state it was
        asked about.
        """
        if z is None:
            if self.needs_memory_state:
                raise ValueError(
                    f"{self.delta.defect_code} carries a memory state, so the "
                    "off-branch query signature is query_clean(x, t, z) (5.1). Pass "
                    "defect_state[t] from the recorded column; there is no value to "
                    "assume for it"
                )
        else:
            z = np.atleast_2d(np.asarray(z, dtype=np.float64))
            if z.shape[0] != n:
                raise ValueError(f"z has {z.shape[0]} frames, states have {n}")

        if WritableField.SENSING_TWIN not in self.wrapper.spec.primary_fields:
            return None
        if z is None:
            # D1: the bias is a level parameter, so the clocks are unread and any
            # value gives the same answer. D10 never reaches here -- it is in
            # MEMORY_DEFECTS and the branch above has already refused.
            clock = np.zeros(n)
            return inject.sensing_offsets(self.delta, clock, clock)
        time, arc = inject.session_clock(z)
        return inject.sensing_offsets(self.delta, time, arc)


@dataclass(frozen=True)
class CTrue:
    """The per-frame ground truth of one batch, with its saturation bookkeeping."""

    c: np.ndarray                 # (N, ACTION_DIM)
    clipped: np.ndarray           # (N,) bool, either side hit the action box
    actions_defective: np.ndarray  # (N, ACTION_DIM) clipped, as stored
    actions_clean: np.ndarray      # (N, ACTION_DIM) clipped, as stored

    @property
    def saturated_share(self) -> float:
        return float(np.mean(self.clipped))

    @property
    def needs_annotation(self) -> bool:
        """9.6.7 item 2: above 5% the cell's Lambda must be annotated."""
        return self.saturated_share > SATURATION_ANNOTATION_SHARE

    def render(self) -> str:
        norm = np.linalg.norm(self.c, axis=1)
        line = (
            f"c_true over {len(self.c)} frames: ||c|| mean {norm.mean():.5f}, "
            f"max {norm.max():.5f}, saturated {100 * self.saturated_share:.2f}%"
        )
        if self.needs_annotation:
            line += (
                f" -- above the {100 * SATURATION_ANNOTATION_SHARE:.0f}% threshold, "
                "this cell's Lambda carries an annotation (9.6.7 item 2)"
            )
        return line


@dataclass(frozen=True)
class TwinQuery:
    """A structurally aligned pair, both sides evaluable at the same (x, t).

    The reference is shared rather than resolved twice. `seed_env` carries no
    defect id (9.1.6), so both members reset to the same object placement and a
    second resolution would produce the same numbers -- but only up to whatever
    the resolver reads from a live model, and 9.6.7 needs the two sides to differ
    in the defect and in nothing else.
    """

    defective: TeacherArm
    clean: TeacherArm

    def __post_init__(self) -> None:
        if not self.clean.delta.is_zero:
            raise ValueError(
                f"the clean side carries {self.clean.delta}; definition 3.1 "
                "differences against the twin at delta = 0"
            )
        if self.defective.reference is not self.clean.reference:
            raise ValueError(
                "the two sides hold different reference objects. 9.6.7 evaluates "
                "both at the same x_ref[t]; two separately resolved references "
                "would put object-placement noise into c_true"
            )

    def c_true(
        self,
        states: np.ndarray,
        t: np.ndarray,
        defect_state: np.ndarray | None = None,
    ) -> CTrue:
        """The 9.6.7 evaluation, batched over frames.

        `states` are the frames the *defective* body visited, which is what makes
        this an off-branch query: the clean twin never reaches them in a rollout
        of its own, and (A2) is the assumption that licenses asking anyway.
        """
        raw_d = self.defective.raw_actions(states, t, defect_state)
        raw_0 = self.clean.raw_actions(states, t)
        a_d = self.defective.tracker.clip_action(raw_d)
        a_0 = self.clean.tracker.clip_action(raw_0)
        return CTrue(
            c=a_d - a_0,
            clipped=np.any(np.abs(raw_d) > 1.0, axis=1) | np.any(np.abs(raw_0) > 1.0, axis=1),
            actions_defective=a_d,
            actions_clean=a_0,
        )


def query_clean(
    embodiment: TeacherArm | TwinQuery,
    states: np.ndarray,
    t: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """The sole entry point for the clean twin's action at off-branch frames.

    This is the fifth and sixth lines of the 9.6.7 listing: the clean twin's
    controller plus its converged feedforward, evaluated at the states the
    defective body visited and at the control step those frames carry. Passing a
    `TwinQuery` reads its clean side, so a caller holding a pair cannot query the
    wrong member by writing the wrong attribute.

    Returns actions already clipped to [-1, 1]. Every consumer of the ground
    truth differences two of these, and 9.6.7 item 2 wants the clip to have
    happened on each side before that subtraction; returning unclipped values
    here would leave the ordering to whoever calls it, which is exactly the
    failure mode the item warns about.

    `z` is the wrapper memory state of 5.1. It is accepted on this signature so
    that the entry point is the same one D2, D4 and D10 use; on the clean twin
    it is `None`, because delta = 0 has no memory to restore.
    """
    arm = embodiment.clean if isinstance(embodiment, TwinQuery) else embodiment
    return arm.actions(states, t, z)


def _clean_arm(embodiment: TeacherArm | TwinQuery) -> TeacherArm:
    arm = embodiment.clean if isinstance(embodiment, TwinQuery) else embodiment
    if not arm.delta.is_zero:
        raise ValueError(
            f"a_dot is the derivative of pi_{{e,0,tau}} and is taken on the clean "
            f"twin (5.1); this arm carries {arm.delta}"
        )
    return arm


def _derivative_frames(
    arm: TeacherArm, states: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The (x, t) batch, refused when a frame has no successor phase.

    a_dot differences the teacher between control steps t and t + 1, so the last
    step of the horizon cannot carry one. 9.4.2's reference finishes stationary,
    which makes holding x_ref past the end look harmless, but the feedforward
    does not finish stationary and the held value would be an invented
    continuation of the demonstration -- so the frame is refused and the caller
    drops it.
    """
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    steps = np.atleast_1d(np.asarray(t, dtype=np.int64))
    last = arm.reference.horizon - 1
    if steps.size and int(steps.max()) >= last:
        raise ValueError(
            f"control step {int(steps.max())} has no successor: the reference ends "
            f"at step {last}, and a_dot is the forward difference between t and "
            "t + 1. Evaluate the field on frames 0 .. T - 2 rather than holding "
            "the reference past its end"
        )
    return states, steps


def a_dot_nominal(
    embodiment: TeacherArm | TwinQuery,
    states: np.ndarray,
    t: np.ndarray,
    *,
    caliber: adot.DelayCaliber,
    actuation: np.ndarray,
    method: adot.DerivativeMethod = adot.DerivativeMethod.FLOW,
) -> np.ndarray:
    """5.1's a_dot for this teacher: (N, nq+nv) and (N,) in, (N, 6) out.

    The adapter between the teacher and `defects.adot`. It exists because the
    two ends disagree about what a policy is: 5.1 writes pi_{e,0,tau}(x) and the
    teacher of 9.4.2 is OSC(x; x_ref[t]) + u_inf[t], a function of the control
    step as well. 9.6.7 settles which one wins -- the query signature is
    (x, t) -- and `adot.Policy` is written on two arguments for that reason, so
    what is left here is binding the reference and the feedforward to a body,
    which is exactly what `TeacherArm` is.

    Both terms of the derivative are kept: the teacher's action changes because
    the state flows and because the phase advances, and 9.2.2's closed form
    c = ell * a_dot^nom comes from advancing the emitted action *sequence* by
    ell, which is the sum of the two. `a_dot_split` reports them separately.

    `caliber` is passed through rather than assumed. The teacher's own delay is
    action-side by construction -- it is the wrapper's control-step ring buffer
    -- but A-23's exclusion is asserted at one place, and routing around that
    place is how a torque-side number would get into a table.

    `actuation` is `TrackingResult.actuation` at those frames, the gripper
    accumulator `defects.adot.ClosedLoopDerivative.flow` cannot do without. It
    travels with the state, which extends 9.6.7 item 1's "one state, two models"
    to it: the accumulator is driven by sign(a_gripper) alone, and a_gripper on
    both members of a pair is the same reference schedule, so the trace recorded
    on the defective body is the clean twin's trace too rather than an
    approximation of it.
    """
    arm = _clean_arm(embodiment)
    states, steps = _derivative_frames(arm, states, t)
    derivative = adot.ClosedLoopDerivative(arm.wrapper, arm.as_policy())
    full = derivative.a_dot(
        states, steps, caliber=caliber, actuation=actuation, method=method
    )
    return full[:, :TASK_CHANNELS]


def a_dot_split(
    embodiment: TeacherArm | TwinQuery,
    states: np.ndarray,
    t: np.ndarray,
    *,
    caliber: adot.DelayCaliber,
    actuation: np.ndarray,
) -> adot.DerivativeSplit:
    """a_dot resolved into grad_x a . f and da/dt, on the six task channels.

    The reason the split is a deliverable and not a debugging aid: 5.1's boxed
    form names only the first term, because it is written for a stationary
    policy where the second does not exist. On this teacher it does, and its
    size decides whether D2's ground truth is the vector the ILC feedforward is
    compared against or a different one. Reported, not asserted.
    """
    arm = _clean_arm(embodiment)
    states, steps = _derivative_frames(arm, states, t)
    derivative = adot.ClosedLoopDerivative(arm.wrapper, arm.as_policy())
    split = derivative.split(states, steps, caliber=caliber, actuation=actuation)
    return adot.DerivativeSplit(
        convective=split.convective[:, :TASK_CHANNELS],
        explicit=split.explicit[:, :TASK_CHANNELS],
        total=split.total[:, :TASK_CHANNELS],
    )


def arm_from_wrapper(
    wrapper: Any,
    reference: ReferenceTrajectory,
    u_inf: np.ndarray,
    *,
    tracker: OSCTracker | None = None,
) -> TeacherArm:
    """Bind a `DefectWrapper` to the reference and feedforward that drove it."""
    tracker = tracker or OSCTracker()
    env = wrapper.env
    base_pos, base_rot = tracker.base_frame(env)
    model = env.sim.model._model
    return TeacherArm(
        wrapper=wrapper,
        reference=reference,
        u_inf=np.asarray(u_inf, dtype=np.float64),
        tracker=tracker,
        site_id=tracker.grip_site_id(env),
        base_pos=base_pos,
        base_rot=base_rot,
        _model=model,
        _data=mujoco.MjData(model),
        _qpos_index=np.asarray(wrapper.index.qpos_index, dtype=np.intp),
    )


def twin_query(
    pair: TwinPair,
    reference: ReferenceTrajectory,
    *,
    u_inf_defective: np.ndarray,
    u_inf_clean: np.ndarray,
    tracker: OSCTracker | None = None,
) -> TwinQuery:
    """Assemble both sides of 9.6.7 from a `defects.inject.build_twin_pair` result."""
    if reference.horizon != HORIZON:
        raise ValueError(f"reference has {reference.horizon} steps, 9.6.1 fixes {HORIZON}")
    tracker = tracker or OSCTracker()
    return TwinQuery(
        defective=arm_from_wrapper(pair.defective, reference, u_inf_defective, tracker=tracker),
        clean=arm_from_wrapper(pair.clean, reference, u_inf_clean, tracker=tracker),
    )
